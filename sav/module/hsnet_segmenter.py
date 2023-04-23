import torch
import torch.nn as nn
import pandas as pd
import pytorch_lightning as pl
from .base_module import BaseModule
from .hsnet.hsnet import HypercorrSqueezeNetwork
from sav.utils.visual import log_fig, plot_evaluation

class HSNetSegmenter(BaseModule):
    def __init__(self, 
                 backbone:str='vgg16', 
                 use_original_imgsize:bool=False,
                 optimizer:str='adam', 
                 learning_rate:float=1e-4, 
                 weight_decay:float=1e-5
                ):
        super().__init__(backbone, optimizer, learning_rate, weight_decay)

        self.use_original_imgsize = use_original_imgsize
        self.hsnet = HypercorrSqueezeNetwork(backbone=backbone, use_original_imgsize=use_original_imgsize)
        
        # freeze the weights and biases in the backbone model
        for param in self.hsnet.backbone.parameters():
            param.requires_grad = False
    
    def forward(self, query_img, support_img, support_mask):
        return self.hsnet(query_img, support_img, support_mask)
    
    def training_step(self, train_batch, batch_idx):
        query_img      = torch.tile(train_batch['query_img'],(1,3,1,1))        # size = [bs,3,512,512]
        query_annot    = train_batch['query_annot']                            # size = [bs,1,512,512]
        support_img   = torch.tile(train_batch['support_imgs'].squeeze(1),(1,3,1,1))      # size = [bs,nshot=1,1,512,512] -> [bs,3,512,512]
        support_annot = train_batch['support_annots'].squeeze()   # size = [bs,nshot=1,1,512,512] -> [bs,512,512]
        specimen_phase_name = train_batch['specimen_phase_name'] 
        
        query_annot_hat = self(query_img, support_img, support_annot)
        
        loss = self.criterion(query_annot_hat, query_annot)
        iou = self.calculate_IoU(query_annot_hat, query_annot)

        metrics = {
            'loss':loss,
            'iou':iou,
        }

        self.log('train/loss', metrics['loss'], prog_bar=True, on_step=True)
        self.log('train/iou', metrics['iou'], prog_bar=True, on_step=True)

        return metrics
    
    def validation_step(self, val_batch, batch_idx):
        query_img      = torch.tile(val_batch['query_img'],(1,3,1,1))        # size = [bs,3,512,512]
        query_annot    = val_batch['query_annot']                            # size = [bs,3,512,512]
        support_img   = torch.tile(val_batch['support_imgs'].squeeze(1),(1,3,1,1))      # size = [bs,nshot=1,1,512,512] -> [bs,3,512,512]
        support_annot = val_batch['support_annots'].squeeze()   # size = [bs,nshot=1,1,512,512] -> [bs,512,512]
        specimen_phase_name = val_batch['specimen_phase_name'] 
        
        query_annot_hat = self(query_img, support_img, support_annot)
        
        loss = self.criterion(query_annot_hat, query_annot)
        iou = self.calculate_IoU(query_annot_hat, query_annot)
        
        metrics = {
            'val_loss':loss,
            'val_iou': iou,
            'class': specimen_phase_name[0],
            'query_img':query_img,
            'query_annot':query_annot,
            'query_annot_hat':query_annot_hat,
        }

        self.log('val/val_loss', metrics['val_loss'], prog_bar=False, on_step=True)
        self.log('val/val_iou', metrics['val_iou'], prog_bar=False, on_step=True)

        return metrics
    
    def validation_epoch_end(self, valid_step_outputs):
        epoch_val_loss = torch.stack([x['val_loss'] for x in valid_step_outputs]).mean()
        epoch_val_iou = torch.stack([x['val_iou'] for x in valid_step_outputs]).mean()
        self.log('val/epoch_val_loss', epoch_val_loss, prog_bar=True, on_step=False)
        self.log('val/epoch_val_iou', epoch_val_iou, prog_bar=True, on_step=False)
        
        # calculate iou per class
        class_iou = {}
        for output in valid_step_outputs:
            if output['class'] not in class_iou.keys():
                class_name = output['class']
                class_iou[class_name] = [output['val_iou']]
            else:
                class_iou[class_name].append(output['val_iou'])
        
        for class_name in class_iou.keys():
            class_iou[class_name] = torch.stack(class_iou[class_name]).mean().detach().cpu().numpy()
            self.log(f'metric/mean_iou_{class_name}', float(class_iou[class_name]))
        
        metrics_list = list()
        for i, class_name in enumerate(class_iou.keys()):
            metrics_list.append([class_name, class_iou[class_name]])
        
        self.logger.log_table(key=f'metrics_per_class_{self.current_epoch}', 
                              columns=['class_name', 'mean_iou'], 
                              data=metrics_list)
        
        self.metrics = pd.DataFrame.from_dict(class_iou, orient='index', columns=['mean_iou'])

        # plot validation results
        last_batch = valid_step_outputs[-1]
        last_batch['query_img'] = last_batch['query_img'][:,0,:,:]
        fig = plot_evaluation(last_batch)
        
        log_fig(fig=fig,
                log_name='visual/eval_check',  
                logger=self.logger, 
                current_epoch=self.current_epoch)