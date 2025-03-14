import os
import torch
import numpy as np
import torchvision
import torchvision.transforms.v2 as transforms
import pytorch_lightning as pl

from pathlib import Path
from typing import Union, Dict, Tuple, AnyStr
from torch.utils.data import random_split, Dataset, DataLoader
from sav.utils.utils import get_img_sizes, sampling_from_dir, sample_paired_img_annot

torchvision.disable_beta_transforms_warning()

class DatamoduleSAV(pl.LightningDataModule):
    def __init__(self,
                 datapath:Union[Path, AnyStr], 
                 nshot:int=5,
                 nsamples:int=1000,
                 contrast:Tuple[float,float]=(0.5,1.2),
                 vflip_p:float=0.5, 
                 hflip_p:float=0.5,
                 rotation_degrees:float=90.0,
                 scale:Tuple[float,float]=(0.8,1.2),
                 crop_size:float=512,
                 copy_paste_prob:float=0.15,
                 val_data_ratio:float=0.15,
                 batch_size:int=20,
                 n_cpu:int=4,
                ):
        super().__init__()
        
        self.nshot = nshot
        self.val_data_ratio = val_data_ratio
        self.batch_size = batch_size
        self.n_cpu = n_cpu
        self.img_sizes = get_img_sizes(datapath)# e.g., {'cauliflower':(2048,1536),'apollo_70017':(1004,1024)}
        self.dataset_full = DatasetSAV(datapath, nshot, nsamples, self.img_sizes, contrast, 
                                       vflip_p, hflip_p, rotation_degrees, scale, crop_size, copy_paste_prob)
    
    def setup(self, stage = None):
        if stage == "fit" or stage is None:
            all_size = self.dataset_full.__len__()
            val_size = int(all_size*self.val_data_ratio) if (self.val_data_ratio > 0.0) else int(all_size*0.15)
            self.dataset_train, self.dataset_val = random_split(self.dataset_full, [(all_size - val_size), val_size])

        if stage == "test":
            all_size = self.dataset_full.__len__()
            val_size = int(all_size*self.val_data_ratio) if (self.val_data_ratio > 0.0) else int(all_size*0.15)
            self.dataset_train, self.dataset_test = random_split(self.dataset_full, [(all_size - val_size), val_size])


    def train_dataloader(self):
        return DataLoader(self.dataset_train,
                          batch_size=self.batch_size,
                          shuffle=True,
                          num_workers=self.n_cpu,
                          pin_memory=True)
    
    def val_dataloader(self):
        return DataLoader(self.dataset_val,
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=self.n_cpu,
                          pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.dataset_test,
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=self.n_cpu,
                          pin_memory=True)
    
    
class DatamoduleDeepLabV3(pl.LightningDataModule):
    def __init__(self,
                 datapath:Union[Path, AnyStr], 
                 nsamples:int=1000,
                 contrast:Tuple[float,float]=(0.5,1.2),
                 vflip_p:float=0.5, 
                 hflip_p:float=0.5,
                 rotation_degrees:float=90.0,
                 scale:Tuple[float,float]=(0.8,1.2),
                 crop_size:float=512,
                 copy_paste_prob:float=0.15,
                 val_data_ratio:float=0.15,
                 batch_size:int=20,
                 n_cpu:int=4,
                ):
        super().__init__()
        self.val_data_ratio = val_data_ratio
        self.batch_size = batch_size
        self.n_cpu = n_cpu
        self.img_sizes = list(get_img_sizes(datapath).values())[0] # get the first value from a dict
        self.dataset_full = DatasetDeepLabV3(datapath, nsamples, self.img_sizes, contrast, vflip_p,
                                             hflip_p, rotation_degrees, scale, crop_size, copy_paste_prob)

    def setup(self, stage = None):
        if stage == "fit" or stage is None:
            all_size = self.dataset_full.__len__()
            val_size = int(all_size*self.val_data_ratio) if (self.val_data_ratio > 0.0) else int(all_size*0.15)
            self.dataset_train, self.dataset_val = random_split(self.dataset_full, [(all_size - val_size), val_size])

        if stage == "test":
            all_size = self.dataset_full.__len__()
            val_size = int(all_size*self.val_data_ratio) if (self.val_data_ratio > 0.0) else int(all_size*0.15)
            self.dataset_train, self.dataset_test = random_split(self.dataset_full, [(all_size - val_size), val_size])


    def train_dataloader(self):
        return DataLoader(self.dataset_train,
                          batch_size=self.batch_size,
                          shuffle=True,
                          num_workers=self.n_cpu,
                          pin_memory=True)
    
    def val_dataloader(self):
        return DataLoader(self.dataset_val,
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=self.n_cpu,
                          pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.dataset_test,
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=self.n_cpu,
                          pin_memory=True)
                          
class DatasetSAV(Dataset):
    def __init__(self, 
                 datapath:Union[Path, AnyStr], 
                 nshot:int=5,
                 nsamples:int=1000,
                 img_sizes:Dict[str, Tuple[int, int]]=None,
                 contrast:Tuple[float,float]=(0.5,1.2),
                 vflip_p:float=0.5, 
                 hflip_p:float=0.5,
                 rotation_degrees:float=90.0,
                 scale:Tuple[float,float]=(0.8,1.2),
                 crop_size:float=512,
                 copy_paste_prob:float=0.15,
                ):
        """
        Args:
            datapath (path): A folder directory containing multiple specimens and phases. The directory structure is as following:
            
                datapath/
                └── specimen/            
                    ├── phase0/
                    │   ├── annotation/        # target masks
                    │   └── image/             # input images
                    ├── phase1/
                    │   ├── annotation/
                    │   └── image/
                    └── ...
            
            shot (int): number of shots in each episode.
            nsamples (int): Number of training images cropped from the original images.
            rotation_degrees (float): roataion angle of the image for the data augmentation
            crop_size (int): size to be cropped from the rotated images
            transform (torchvision.transforms): the transformation to torch.tensor (preprocessing).
        """
        super().__init__()
        self.base_path = datapath
        self.nshot = nshot
        self.nsamples = nsamples
        self.copy_paste_prob = copy_paste_prob
        
        self.specimen_names = [sample for sample in os.listdir(self.base_path) if '.' not in sample]
        
        self.random_rotation_crop = {}
        for key in img_sizes.keys():
            self.random_rotation_crop[key]=RandomRotationCrop(img_sizes[key],vflip_p, hflip_p,rotation_degrees, scale, crop_size)
        self.normalize = transforms.Compose([transforms.ColorJitter(contrast=contrast),
                                             transforms.Normalize(mean=[0.5],std=[0.5])])
        
        self.image_copy_paste = {}
        for key in img_sizes.keys():
            self.image_copy_paste[key]=ImageCopyPaste(base_path=self.base_path, 
                                                      transform=self.random_rotation_crop[key],
                                                      normalize=self.normalize,
                                                      p=self.copy_paste_prob)
    
    def __len__(self):
        return self.nsamples
    
    def __getitem__(self, idx, phase_dir=None): # If we only want to sample a specific phase specify the phase_dir
        # ignores idx during training & testing and perform uniform sampling over object classes to form an episode

        # Randomly sample a phase from all specimens
        if phase_dir is None:
            specimen_dir = sampling_from_dir(self.base_path)
            phase_dir = sampling_from_dir(specimen_dir)
            specimen_phase_name = specimen_dir.split('/')[-1] + '_' + phase_dir.split('/')[-1]
        else:
            specimen_phase_name = phase_dir.split('/')[-2] + '_' + phase_dir.split('/')[-1]
        
        specimen_name = specimen_phase_name.split('_')[0]
        
        # Get the querry img and annot
        query_img, query_annot = sample_paired_img_annot(phase_dir)
        # Get the rotated and cropped img and annot
        query_transformed = self.random_rotation_crop[specimen_name](query_img, query_annot)
        query_img, query_annot = query_transformed['img'], query_transformed['annot']
        query_img = self.normalize(query_img) # size = [1,512,512]

        # apply copy_paste
        query_img, query_annot = self.image_copy_paste[specimen_name](query_img, query_annot, specimen_phase_name)
        
        # Get the support set
        support_imgs = []
        support_annots = []
        for i in range(self.nshot):
            support_img, support_annot = sample_paired_img_annot(phase_dir)
            support_transformed = self.random_rotation_crop[specimen_name](support_img, support_annot)
            support_img, support_annot = support_transformed['img'], support_transformed['annot']
            support_img = self.normalize(support_img)
            # apply copy_paste
            support_img, support_annot = self.image_copy_paste[specimen_name](query_img, query_annot, specimen_phase_name)
            support_imgs.append(support_img)
            support_annots.append(support_annot)
        
        return {'query_img':query_img.to(dtype=torch.float32),                               # size = [1,1,512,512]
                'query_annot':query_annot.to(dtype=torch.float32),                           # size = [1,1,512,512]
                'support_imgs':torch.stack(support_imgs,dim=0).to(dtype=torch.float32),      # size = [nshot,1,512,512]
                'support_annots': torch.stack(support_annots,dim=0).to(dtype=torch.float32), # size = [nshot,1,512,512]
                'specimen_phase_name':specimen_phase_name,
               }


class DatasetDeepLabV3(Dataset):
    def __init__(self, 
                 datapath:Union[Path, AnyStr], 
                 nsamples:int=1000,
                 img_sizes:Dict[str, Tuple[int, int]]=None,
                 contrast:Tuple[float,float]=(0.5,1.2),
                 vflip_p:float=0.5, 
                 hflip_p:float=0.5,
                 rotation_degrees:float=90.0,
                 scale:Tuple[float,float]=(0.8,1.2),
                 crop_size:float=512,
                 copy_paste_prob:float=0.15,
                ):
        """
        Args:
            datapath (path): A folder directory containing multiple specimens and phases. The directory structure is as following:
            
                datapath/        
                └── phase/
                    ├── annotation/        # target masks
                    └── image/             # input images
            
            nsamples (int): Number of training images cropped from the original images.
            rotation_degrees (float): roataion angle of the image for the data augmentation
            crop_size (int): size to be cropped from the rotated images
            transform (torchvision.transforms): the transformation to torch.tensor (preprocessing).
        """
        super().__init__()
        self.base_path = datapath
        self.nsamples = nsamples
        self.copy_paste_prob = copy_paste_prob
    
        
        self.random_rotation_crop=RandomRotationCrop(img_sizes,vflip_p, hflip_p,rotation_degrees, scale, crop_size)
        self.normalize = transforms.Compose([transforms.ColorJitter(contrast=contrast),
                                             transforms.Normalize(mean=[0.5],std=[0.5])])
        self.image_copy_paste=ImageCopyPaste(base_path=self.base_path, 
                                                      transform=self.random_rotation_crop,
                                                      normalize=self.normalize,
                                                      p=self.copy_paste_prob)
    
    def __len__(self):
        return self.nsamples
    
    def __getitem__(self, idx, phase_dir=None): # If we only want to sample a specific phase specify the phase_dir
        
        if phase_dir==None:
            specimen_dir = sampling_from_dir(self.base_path)
            phase_dir = sampling_from_dir(specimen_dir)
            specimen_phase_name = specimen_dir.split('/')[-1] + '_' + phase_dir.split('/')[-1]
                                             
                                             
        # Get the querry img and annot
        query_img, query_annot = sample_paired_img_annot(phase_dir)
        # Get the rotated and cropped img and annot
        query_transformed = self.random_rotation_crop(query_img, query_annot)
        query_img, query_annot = query_transformed['img'], query_transformed['annot']
        query_img = self.normalize(query_img) # size = [1,512,512]

        # apply copy_paste
        query_img, query_annot = self.image_copy_paste(query_img, query_annot, specimen_phase_name)
        
        return {'query_img':query_img.to(dtype=torch.float32),                               # size = [1,1,512,512]
                'query_annot':query_annot.to(dtype=torch.float32),                           # size = [1,1,512,512]
                'specimen_phase_name':specimen_phase_name,
               }

class RandomRotationCrop:
    """
    Args:
        rotation_degrees (float): roataion angle of the image for the data augmentation
        crop_size (int): size to be cropped from the rotated images
    Return:
        (torch.tensor): a dictionary containing img (size=[h,w]) and annot (size=[h,w])
    """
    def __init__(self, img_size:Tuple[int,int],vflip_p:float, hflip_p:float,rotation_degrees:float, scale:Tuple[float,float], crop_size:int):
        self.random_rotation_crop = transforms.Compose([transforms.ToImageTensor(), 
                                                        transforms.ConvertImageDtype(torch.float32), # size = [2, height, width]
                                                        # transforms.Resize(size=resize),
                                                        transforms.ScaleJitter(img_size,scale_range=scale),
                                                        transforms.RandomVerticalFlip(p=vflip_p),
                                                        transforms.RandomHorizontalFlip(p=hflip_p),
                                                        transforms.RandomRotation(degrees=rotation_degrees),
                                                        transforms.RandomCrop(size=crop_size, pad_if_needed=True),
                                                       ])
    def __call__(self, img, annot, background_mode=False):
        # Rescale to [0, 1]
        img, annot = np.divide(img, img.max()), np.divide(annot, annot.max())
        # make sure the image and annotation are cropped from the same area
        img_annot = np.stack([img, annot],axis=2) # size = [height, width, 2]
        # img_annot_ = self.random_rotation_crop(img_annot)
        if background_mode:
            img_annot_ = self.random_rotation_crop(img_annot)
        else:
            while True: # keep sampling areas if the ROI isn't cropped from the image (at least including 1% of the total area)
                img_annot_ = self.random_rotation_crop(img_annot)
                if img_annot_[1].sum() > 0.01*img_annot_[1].size(0)*img_annot_[1].size(1):break

        return {'img': img_annot_[0].unsqueeze(0), 'annot': torch.where(img_annot_[1].unsqueeze(0)>0.5,1.0,0.0)} # size = [1,512,512]

class ImageCopyPaste:
    def __init__(self, 
                 base_path:Union[Path, AnyStr], 
                 transform:"torchvision.transforms",
                 normalize:"torchvision.transforms",
                 p=0.5):
        self.base_path = base_path
        self.transform = transform
        self.normalize = normalize
        self.p = p
        
    def __call__(self, 
                 paste_img:"torch.Tensor", 
                 paste_annot:"torch.Tensor", 
                 specimen_phase_name:str
                )->Tuple["torch.Tensor", "torch.Tensor"]:
        
        if np.random.uniform() > self.p:
            return (paste_img, paste_annot)
        else:
            back_phase_dir = os.path.join(self.base_path, *specimen_phase_name.split('_'))
            
            back_img, back_annot = sample_paired_img_annot(back_phase_dir)
            back_img_transformed = self.transform(back_img, back_annot, background_mode=True)
            back_img, back_annot = back_img_transformed['img'], back_img_transformed['annot']
            back_img = self.normalize(back_img) # size = [1,512,512]

            copy_paste_img = self.image_copy_paste(back_img, paste_img, paste_annot)
            copy_paste_annot = self.annot_copy_paste(back_annot, paste_annot)

            return (copy_paste_img, copy_paste_annot)
        
    def image_copy_paste(self, 
                         img:"torch.Tensor", 
                         paste_img:"torch.Tensor", 
                         alpha:"torch.Tensor", 
                        ) -> "torch.Tensor":
        img_dtype = img.dtype
        device = img.device
        img = paste_img * alpha + img * (1 - alpha)
        img = img.to(dtype=img_dtype, device = device)
        return img

    def annot_copy_paste(self,
                         back_annot:"torch.Tensor", 
                         paste_annot:"torch.Tensor", 
                        )-> "torch.Tensor":
        #eliminate pixels that will be pasted over
        return torch.logical_or(back_annot, paste_annot).to(dtype=paste_annot.dtype, device = paste_annot.device)
