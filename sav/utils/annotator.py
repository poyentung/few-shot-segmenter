import os
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from rich.progress import track
from PIL import Image, ImageOps
from torchvision import transforms
from torch.utils.data import DataLoader
from typing import Union, List, Tuple, AnyStr
from sav.module.fs_segmenter import FewShotSegmenter
from sav.utils.utils import downsample_and_pad, unpad_and_upsample, calculate_iou_with_truth

class Annotator:
    def __init__(self, 
                 model: FewShotSegmenter,
                 # dataset: DatasetSAV,
                 # phase: str,
                 transform: "torchvision.transforms"=None,
                 down_sampling:int=2,
                 patch_width:int=256, 
                 patch_height:int=256,
                 margin:int=28,
                 batch_size:int=3,
                 keep_dim:bool=False,
                 save_init_imgs:bool=True,
                 cuda_device=0,
                ):
        
        if torch.cuda.is_available():
            torch.cuda.set_device(cuda_device)
        self.device = "cuda" if torch.cuda.is_available() else model.device
        self.model = model.to(device=self.device)
        
        # self.init_img_height, self.init_img_width = resize
        self.transform = transform if transform is not None else transforms.Compose([transforms.ToTensor(),
                                                                                     transforms.Resize((patch_height+(margin*2), patch_width+(margin*2))),
                                                                                     transforms.Normalize(mean=[0.5],std=[0.5])])
        self.transform_annot = transforms.Compose([transforms.ToTensor(),
                                                   transforms.Resize((patch_height, patch_width))])
        self.patch_width = patch_width
        self.patch_height = patch_height
        self.down_sampling = down_sampling
        self.margin = margin
        self.batch_size = batch_size
        self.keep_dim = keep_dim
        self.save_init_imgs= save_init_imgs
    
    def __call__(self, 
                 query_img_path:Union[Path, List[Path]],
                 support_imgs_dir:Union[Path, AnyStr],
                 support_annots_dir:Union[Path, AnyStr],
                 batch_size:int=None,
                 keep_dim:bool=None,
                 save_dir:Union[Path, AnyStr]=None):
        
        batch_size = self.batch_size if batch_size is None else batch_size
        keep_dim = self.keep_dim if keep_dim is None else keep_dim
        
        # the query_img_path is a single image
        if query_img_path.endswith('.tiff'):
            return self.detect(query_img_path, support_imgs_dir, support_annots_dir, batch_size, keep_dim)
        
        # the query_img_path is folder containing multiple images to be segmented
        else:
            if save_dir is None: 
                raise TypeError('save_dir cannot be None.')
            elif not os.path.exists(save_dir):
                os.makedirs(save_dir)
                
            query_img_paths = sorted([os.path.join(query_img_path, _img_path) for _img_path in os.listdir(query_img_path) if ('.tiff' in _img_path) and (not _img_path.startswith('._'))])
            
            for _img_path in track(query_img_paths, description="[Segementing]"):
                out = self.detect(_img_path, 
                                  support_imgs_dir, 
                                  support_annots_dir,
                                  batch_size,
                                  keep_dim)
                recon_img = Image.fromarray(np.where(out['annot']>0.5,255,0).astype(np.int8)).convert('L')
                resized_img = Image.fromarray(out['raw'])
                
                mask_dir = os.path.join(save_dir,'mask')
                if not os.path.exists(mask_dir):os.makedirs(mask_dir)
                
                img_name = os.path.basename(_img_path).split('.')[0]
                recon_img.save(os.path.join(mask_dir, img_name+'_mask.tiff'))
                
                if self.save_init_imgs:
                    rescale_img_dir = os.path.join(save_dir,'resized_img')
                    if not os.path.exists(rescale_img_dir):os.makedirs(rescale_img_dir)
                    resized_img.save(os.path.join(rescale_img_dir, img_name+'.tiff'))
    
    def detect(self, 
               query_img_path:Union[Path, AnyStr],
               support_imgs_dir:Union[Path, AnyStr],
               support_annots_dir:Union[Path, AnyStr],
               batch_size:int,
               keep_dim:bool=False):
        
        # process padding and resizing the image
        query_imgs_init = Image.open(query_img_path)
        query_imgs_resized = downsample_and_pad(query_imgs_init, (self.patch_width, self.patch_height), self.down_sampling)
        ncol, nrow = query_imgs_resized.size[0]//self.patch_width, query_imgs_resized.size[1]//self.patch_height
        
        query_imgs = self.create_patches(query_imgs_resized, self.patch_width, self.patch_height, self.margin)
        query_imgs = torch.stack([self.transform(img) for img in query_imgs])
        
        support_imgs = self.get_imgs(support_imgs_dir) # return a list of PIL.Image images
        support_imgs = torch.stack([self.transform(img) for img in support_imgs])
        support_imgs = torch.tile(support_imgs, (batch_size,1,1,1,1))
        
        support_annots = self.get_imgs(support_annots_dir)
        support_annots = [np.where(np.array(annot)>0,1,0) for annot in support_annots]
        support_annots = torch.stack([torch.where(self.transform_annot(img)>0.5,1.0,0.0) for img in support_annots])
        support_annots = torch.tile(support_annots, (batch_size,1,1,1,1))
        
        query_annot_hat=[]
        query_imgs = DataLoader(query_imgs, batch_size=batch_size)
        for batch in query_imgs:
            if batch.size(0)!=batch_size:
                support_imgs, support_annots = support_imgs[:batch.size(0)], support_annots[:batch.size(0)]
            batch_annot_hat = self.model(query_img=batch.to(self.device), 
                                         support_imgs=support_imgs.to(self.device), 
                                         support_annots=support_annots.to(self.device)).detach().cpu()
            query_annot_hat.append(batch_annot_hat)
            
        query_annot_hat = torch.cat(query_annot_hat, dim=0)
        query_annot_hat_binary = torch.where(query_annot_hat>0.5,1.0,0.0).squeeze().detach().cpu().numpy()
        recon_img = self.recover_patches(query_annot_hat_binary,
                                         patch_width=self.patch_width, 
                                         patch_height=self.patch_height,
                                         margin=self.margin,
                                         n_row=nrow,
                                         n_col=ncol)
        if keep_dim == False:
            return {'raw': np.array(query_imgs_resized),
                    'annot':recon_img}
        else:
            recon_img = unpad_and_upsample(img=recon_img,
                                           up_sampling=self.down_sampling,
                                           init_size=query_imgs_init.size)
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
        # imgs = patchify(np.array(img), (patch_height+(margin*2),patch_width+(margin*2)),step=patch_height)
        # imgs = [img for img in imgs.reshape(-1,imgs.shape[2],imgs.shape[3])]
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
        # annot_imgs = annot_imgs.reshape(n_row, n_col, patch_height, patch_width)
        # annot_imgs = unpatchify(annot_imgs, (int(n_row*patch_height), int(n_col*patch_height)))
        
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
    
    @classmethod
    def benchmark_iou(cls, 
                      pred_annots_path:Union[Path, AnyStr],
                      truth_annots_path:Union[Path, AnyStr], 
                      img_indices:Tuple[int,int,int,int],
                      save_csv_name:str=None):
        
        iou = calculate_iou_with_truth(pred_annots_path,
                                        truth_annots_path,
                                        img_indices=img_indices)
            
        if save_csv_name is not None: pd.DataFrame(dict(iou=iou)).to_csv(save_csv_name)
        return iou


class AnnotatorDeeplabV3(Annotator):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def __call__(self, 
                 query_img_path:Union[Path, List[Path]],
                 batch_size:int=None,
                 keep_dim:bool=None,
                 save_dir:Union[Path, AnyStr]=None):
        
        batch_size = self.batch_size if batch_size is None else batch_size
        keep_dim = self.keep_dim if keep_dim is None else keep_dim
        
        # the query_img_path is a single image
        if query_img_path.endswith('.tiff'):
            return self.detect(query_img_path, batch_size, keep_dim)
        
        # the query_img_path is folder containing multiple images to be segmented
        else:
            if save_dir is None: 
                raise TypeError('save_dir cannot be None.')
            elif not os.path.exists(save_dir):
                os.makedirs(save_dir)
                
            query_img_paths = sorted([os.path.join(query_img_path, _img_path) for _img_path in os.listdir(query_img_path) if ('.tiff' in _img_path) and (not _img_path.startswith('._'))])
            
            for _img_path in track(query_img_paths, description="[Segementing]"):
                out = self.detect(_img_path, batch_size, keep_dim)
                recon_img = Image.fromarray(np.where(out['annot']>0.5,255,0).astype(np.int8)).convert('L')
                resized_img = Image.fromarray(out['raw'])
                
                mask_dir = os.path.join(save_dir,'mask')
                if not os.path.exists(mask_dir):os.makedirs(mask_dir)
                
                img_name = os.path.basename(_img_path).split('.')[0]
                recon_img.save(os.path.join(mask_dir, img_name+'_mask.tiff'))
                
                if self.save_init_imgs:
                    rescale_img_dir = os.path.join(save_dir,'resized_img')
                    if not os.path.exists(rescale_img_dir):os.makedirs(rescale_img_dir)
                    resized_img.save(os.path.join(rescale_img_dir, img_name+'.tiff'))
    
    def detect(self, 
               query_img_path:Union[Path, AnyStr],
               batch_size:int,
               keep_dim:bool=False):
        
        # process padding and resizing the image
        query_imgs_init = Image.open(query_img_path)
        query_imgs_resized = downsample_and_pad(query_imgs_init, (self.patch_width, self.patch_height), self.down_sampling)
        ncol, nrow = query_imgs_resized.size[0]//self.patch_width, query_imgs_resized.size[1]//self.patch_height
        
        query_imgs = self.create_patches(query_imgs_resized, self.patch_width, self.patch_height, self.margin)
        query_imgs = torch.stack([self.transform(img) for img in query_imgs])
        
        query_annot_hat=[]
        query_imgs = DataLoader(query_imgs, batch_size=batch_size)
        for batch in query_imgs:
            batch_annot_hat = self.model(query_img=batch.to(self.device)).detach().cpu()
            query_annot_hat.append(batch_annot_hat)
            
        query_annot_hat = torch.cat(query_annot_hat, dim=0)
        query_annot_hat_binary = torch.where(query_annot_hat>0.5,1.0,0.0).squeeze().detach().cpu().numpy()
        recon_img = self.recover_patches(query_annot_hat_binary,
                                         patch_width=self.patch_width, 
                                         patch_height=self.patch_height,
                                         margin=self.margin,
                                         n_row=nrow,
                                         n_col=ncol)
        if keep_dim == False:
            return {'raw': np.array(query_imgs_resized),
                    'annot':recon_img}
        else:
            recon_img = unpad_and_upsample(img=recon_img,
                                           up_sampling=self.down_sampling,
                                           init_size=query_imgs_init.size)
            return {'raw': np.array(query_imgs_init),
                    'annot':recon_img}