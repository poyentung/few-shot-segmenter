import os
import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image, ImageOps
from pathlib import Path
from rich.progress import track
from typing import Union, List, Dict, Tuple, AnyStr
from sav.module.fs_segmenter import FewShotSegmenter
from sav.datamodule import DatasetSAV

import matplotlib.pyplot as plt

class Annotator:
    def __init__(self, 
                 model: FewShotSegmenter,
                 # dataset: DatasetSAV,
                 # phase: str,
                 transform: "torchvision.transforms"=None,
                 resize:Tuple[int,int]=(768,1024),
                 patch_width:int=256, 
                 patch_height:int=256,
                 margin:int=28,
                ):
        self.model = model
        self.device = model.device
        # self.dataset = dataset
        # self.phase = phase
        self.init_img_height, self.init_img_width = resize
        self.transform = transform if transform is not None else transforms.Compose([transforms.ToTensor(),
                                                                                     # transforms.Resize((patch_height, patch_width)),
                                                                                     transforms.Resize((patch_height+(margin*2), patch_width+(margin*2))),
                                                                                     transforms.Normalize(mean=[0.5],std=[0.5])])
        self.transform_annot = transforms.Compose([transforms.ToTensor(),
                                                   transforms.Resize((patch_height, patch_width))])
                                                   # transforms.Resize((patch_height+(margin*2), patch_width+(margin*2)))])
        self.patch_width = patch_width
        self.patch_height = patch_height
        self.nrow = self.init_img_height // self.patch_height 
        self.ncol = self.init_img_width // self.patch_width
        self.margin = margin
    
    def __call__(self, 
                 query_img_path:Union[Path, List[Path]],
                 support_imgs_dir:Union[Path, AnyStr],
                 support_annots_dir:Union[Path, AnyStr],
                 save_dir:Union[Path, AnyStr]=None):
        
        if isinstance(query_img_path, list):
            return self.detect(query_img_path, support_imgs_dir, support_annots_dir)
        else:
            if save_dir is None: 
                raise TypeError('save_dir cannot be None.')
            elif not os.path.exists(save_dir):
                os.makedirs(save_dir)
                
            query_img_paths = sorted([os.path.join(query_img_path, _img_path) for _img_path in os.listdir(query_img_path) if '.tiff' in _img_path])
            
            for _img_path in track(query_img_paths, description="Segementing"):
                
                out = self.detect(_img_path, 
                                  support_imgs_dir, 
                                  support_annots_dir)
                recon_img = Image.fromarray(np.where(out['annot']>0.5,255,0).astype(np.int8)).convert('L')
                
                img_name = _img_path.split('/')[-1].split('.')[0]
                recon_img.save(os.path.join(save_dir, img_name+'_mask.tiff'))
    
    def detect(self, 
               query_img_path:Union[Path, AnyStr],
               support_imgs_dir:Union[Path, AnyStr],
               support_annots_dir:Union[Path, AnyStr]):
        
        query_imgs_init = Image.open(query_img_path).resize((self.init_img_width, self.init_img_height))
        query_imgs = self.create_patches(query_imgs_init, self.patch_height, self.patch_width, self.margin)
        query_imgs = torch.stack([self.transform(img) for img in query_imgs])
        # support_data = self.dataset.__getitem__(np.random.choice(len(dataset), 1), self.phase)
        # support_imgs =  torch.tile(support_data['support_imgs'], (len(query_imgs),1,1,1,1))
        # support_annots = torch.tile(support_data['support_annots'], (len(query_imgs),1,1,1,1))
        
        support_imgs = self.get_imgs(support_imgs_dir) # return a list of PIL.Image images
        support_imgs = torch.stack([self.transform(img) for img in support_imgs])
        support_imgs = torch.tile(support_imgs, (query_imgs.size(0),1,1,1,1))
        
        support_annots = self.get_imgs(support_annots_dir)
        support_annots = [np.where(np.array(annot)>0,1,0) for annot in support_annots]
        support_annots = torch.stack([torch.where(self.transform_annot(img)>0.5,1.0,0.0) for img in support_annots])
        support_annots = torch.tile(support_annots, (query_imgs.size(0),1,1,1,1))

        query_annot_hat = self.model(query_img=query_imgs.to(self.device), 
                                     support_imgs=support_imgs.to(self.device), 
                                     support_annots=support_annots.to(self.device))

        # query_annot_hat_binary = query_annot_hat.squeeze().detach().cpu().numpy()
        query_annot_hat_binary = torch.where(query_annot_hat>0.5,1.0,0.0).squeeze().detach().cpu().numpy()
        recon_img = self.recover_patches(query_annot_hat_binary,
                                         patch_width=self.patch_width, 
                                         patch_height=self.patch_height,
                                         margin=self.margin,
                                         n_row=self.nrow,
                                         n_col=self.ncol)
        return {'raw': np.array(query_imgs_init),
                'annot':recon_img}
    
    @classmethod
    def get_imgs(cls, img_dir:Union[Path, AnyStr])-> List["PIL.Image"]:
        img_names = sorted([img_name for img_name in os.listdir(img_dir) if '.ipynb' not in img_name])
        imgs = []
        for img_name in img_names:
            img = Image.open(os.path.join(img_dir, img_name))
            imgs.append(img)
        return imgs
    
    @classmethod
    def create_patches(cls,
                       img:"PIL.Image", 
                       patch_width:int=256, 
                       patch_height:int=256,
                       margin:int=0,
                      ) -> List["numpy.ndarray"]:

        # Get the size of the image
        width, height = img.size

        img = ImageOps.expand(image=img, border=margin)
        
        imgs = []
        # Loop through the image and create patches
        for i in range(0, width, patch_width):
            for j in range(0, height, patch_height):
                # Define the coordinates of the patch
                left = i
                top = j 
                right = i + patch_width + (margin*2) 
                bottom = j + patch_height + (margin*2) 

                # Create a new image object for the patch
                patch = np.array(img.crop((left, top, right, bottom))).astype(np.uint8)
                imgs.append(patch)
        return imgs
    
    @classmethod
    def recover_patches(cls,
                        annot_imgs:"numpy.ndarray",
                        patch_width:int=256, 
                        patch_height:int=256,
                        margin:int=0,
                        n_row:int=3,
                        n_col:int=4
                       )->"numpy.ndarray":
        
        annot_imgs = annot_imgs[..., margin:patch_height+margin, margin:patch_width+margin]
        full_width= n_col*patch_width
        full_height= n_row*patch_height

        # Create a new image object for the reconstructed image
        reconstructed_image = Image.new('L', (full_width, full_height))

        # Loop through the patches and paste them into the reconstructed image
        for i in range(0, n_col):
            for j in range(0, n_row):
                # Open the patch file
                patch = Image.fromarray(annot_imgs[i*n_row+j])
                # Paste the patch into the reconstructed image
                reconstructed_image.paste(patch, (i*patch_width, j*patch_height))

        return np.array(reconstructed_image)