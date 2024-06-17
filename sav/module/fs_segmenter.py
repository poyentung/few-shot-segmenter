import torch
import torch.nn as nn
from torchvision.models import vgg, vit_b_16
from torchvision.models.segmentation import (
    deeplabv3_resnet50, 
    deeplabv3_resnet101,
    deeplabv3_mobilenet_v3_large
)
from .loss import FocalDiceLoss
from .base_module import (
    BaseModule, 
    DeepLabV3Module,
    set_greyscale_weights, 
    TimeDistributed, 
    GlobalAveragePooling2D
)
class FewShotSegmenter(BaseModule):
    def __init__(self, 
                 backbone:str='vgg16', 
                 optimizer:str='adam', 
                 learning_rate:float=1e-4, 
                 weight_decay:float=1e-5
                ):
        super().__init__(backbone, optimizer, learning_rate, weight_decay)
        
        self.criterion = FocalDiceLoss(lmbda=0.9)
        
        def building_blocks_trans(in_dim, out_dim, filter_size=3,stride=1,padding=0):
            return nn.Sequential(nn.ConvTranspose2d(in_dim, out_dim, filter_size, stride=stride, padding=padding),
                                 nn.InstanceNorm2d(out_dim),
                                 nn.ReLU())
        
        # Backbone network initialization
        if backbone == 'vgg16':
            backbone = vgg.vgg16(pretrained=True)
            backbone = set_greyscale_weights(backbone)
            block1 = list(backbone.children())[0][:5]         # 64 filters
            block2 = list(backbone.children())[0][5:10]       # 128 filters
            block3 = list(backbone.children())[0][10:16]      # 256 filters (remove maxpool2d)
            block4 = list(backbone.children())[0][17:23]      # 512 filters (remove maxpool2d)
            self.backbone_encoder = nn.Sequential(*block1, *block2, *block3, *block4)
            
            # freeze the weights and biases in the backbone model
            # for param in self.backbone_encoder.parameters():
            #     param.requires_grad = False
                
            self.conv_s = nn.Sequential(nn.Conv2d(512,128,3,1,1), nn.ReLU())
            self.conv_q = nn.Sequential(nn.Conv2d(512,128,3,1,1), nn.ReLU())
            
            self.time_distributed1 = TimeDistributed(self.backbone_encoder)
            self.time_distributed2 = TimeDistributed(self.conv_s)
            
            self.common_rep_conv = nn.Sequential(nn.Conv2d(256,128,3,1,1), nn.InstanceNorm2d(128), nn.ReLU())
            
            self.global_avgpool = GlobalAveragePooling2D()
            
            self.decoder = nn.Sequential(building_blocks_trans(128,128,3,1,1),
                                         nn.UpsamplingNearest2d(scale_factor=2),
                                         building_blocks_trans(128,128,3,1,1),
                                         nn.UpsamplingNearest2d(scale_factor=2),
                                         building_blocks_trans(128,128,3,1,1),
                                         nn.Conv2d(128,64,3,1,1),
                                         nn.Conv2d(64,2,3,1,1),
                                         nn.Conv2d(2,1,1,1,0))
            
        elif backbone == 'vit_b_16':
            self.backbone = vit_b_16()
            self.backbone_encoder = nn.Sequential(*list(self.backbone.children())[:-1])[1]
            self.conv_s = nn.Sequential(nn.Conv2d(768,128,3,1,1), nn.ReLU())
            self.conv_q = nn.Sequential(nn.Conv2d(768,128,3,1,1), nn.ReLU())
            self.time_distributed2 = TimeDistributed(self.conv_s)
            
            for param in self.backbone.parameters():
                param.requires_grad = False
            
            self.common_rep_conv = nn.Sequential(nn.Conv2d(256,128,3,1,1), nn.InstanceNorm2d(128), nn.ReLU())
            
            self.global_avgpool = GlobalAveragePooling2D()

            self.decoder = nn.Sequential(building_blocks_trans(128,128,3,1,1),
                                         nn.UpsamplingNearest2d(scale_factor=2),
                                         building_blocks_trans(128,128,3,1,1),
                                         nn.UpsamplingNearest2d(scale_factor=2),
                                         building_blocks_trans(128,128,3,1,1),
                                         nn.UpsamplingNearest2d(scale_factor=2),
                                         building_blocks_trans(128,128,3,1,1),
                                         nn.UpsamplingNearest2d(scale_factor=2),
                                         building_blocks_trans(128,128,3,1,1),
                                         nn.Conv2d(128,64,3,1,1),
                                         nn.Conv2d(64,2,3,1,1),
                                         nn.Conv2d(2,1,1,1,0))
        else:
            raise Exception(f'Unavailable backbone: {backbone}')
        
    def forward(self, query_img, support_imgs, support_annots):
        
        if self.backbone_type == 'vit_b_16':
            vit_out = self.vit_forward(query_img, support_imgs)
            query_img_encoded    = vit_out['query_img_encoded']
            support_imgs_encoded = vit_out['support_imgs_encoded']
            
        elif self.backbone_type == 'vgg16':
            # Extract features using the pre-trained VGG model
            # with torch.no_grad():
            query_img_encoded    = self.backbone_encoder(query_img)
            support_imgs_encoded = self.time_distributed1(support_imgs)
        
        # Pass the features to the corresponding Conv2d layers for supports and query images
        support_imgs_encoded = self.time_distributed2(support_imgs_encoded)
        query_img_encoded    = self.conv_q(query_img_encoded)
        
        # Global Representation
        support_imgs_encoded  = self.global_avgpool(support_annots, support_imgs_encoded) 
        # support_imgs_encoded = torch.tile(support_imgs_encoded, dims=(query_img_encoded.size(0),1,1,1))
        
        # concaternate [support_imgs_encoded, query_img_encoded] along dim = c
        common_rep = torch.cat([support_imgs_encoded, query_img_encoded], dim=1) 
        out = self.common_rep_conv(common_rep) # size = [bs,128*2,w',h']
        
        # Decode to query segment
        out = self.decoder(out)
        return out
    
    
    def vit_forward(self, query_img, support_imgs):
        query_img = torch.tile(query_img, (1,3,1,1))
        query_img = self.backbone._process_input(query_img)
        
        # Expand the class token to the full batch
        batch_class_token = self.backbone.class_token.expand(query_img.size(0), -1, -1)
        query_img = torch.cat([batch_class_token, query_img], dim=1)
        query_img = self.backbone_encoder(query_img)[:,1:]
        query_img_encoded = query_img.view(-1,14,14,768).permute(0,3,1,2) # size = [bs,768,14,14]
        
        support_imgs = torch.tile(support_imgs, (1,1,3,1,1))
        support_imgs_encoded = []
        for support_img in support_imgs:
            support_img = self.backbone._process_input(support_img)
        
            # Expand the class token to the full batch
            batch_class_token = self.backbone.class_token.expand(support_img.size(0), -1, -1)
            support_img = torch.cat([batch_class_token, support_img], dim=1)
            support_img = self.backbone_encoder(support_img)[:,1:]
            support_img_encoded = support_img.view(-1,14,14,768).permute(0,3,1,2) # size = [nshot,768,14,14]
            support_imgs_encoded.append(support_img_encoded)
        
        return {'query_img_encoded'   :query_img_encoded,                 # size = [bs,768,14,14]
                'support_imgs_encoded':torch.stack(support_imgs_encoded)} # size = [bs,3,768,14,14]
    
class DeepLabV3(DeepLabV3Module):
    def __init__(self, 
                 backbone:str='deeplabv3_resnet50', 
                 optimizer:str='adam', 
                 learning_rate:float=1e-4, 
                 weight_decay:float=1e-5
                ):
        super().__init__(backbone, optimizer, learning_rate, weight_decay)
        
        self.criterion = FocalDiceLoss(lmbda=0.9)
        
        # Backbone network initialization
        if backbone == 'deeplabv3_resnet50':
            self.model = deeplabv3_resnet50()
            self.model.backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.model.classifier[4] = nn.Conv2d(256, 1, kernel_size=1, stride=1)
            
        elif backbone == 'deeplabv3_resnet101':
            self.model = deeplabv3_resnet101()
            self.model.backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.model.classifier[4] = nn.Conv2d(256, 1, kernel_size=1, stride=1)
            
        elif backbone == 'deeplabv3_mobilenet_v3_large':
            self.model = deeplabv3_mobilenet_v3_large()
            self.model.backbone['0'][0] = nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1, bias=False)
            self.model.classifier[4] = nn.Conv2d(256, 1, kernel_size=1, stride=1)
            
        else:
            raise Exception(f'Unavailable backbone: {backbone}')
        
    def forward(self, query_img):
        return self.model(query_img)['out']