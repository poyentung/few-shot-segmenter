import os
import numpy as np
import PIL.Image as Image
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Union, AnyStr

def make_binary_mask_dir(load_dir:Union[Path, AnyStr],
                         save_dir:Union[Path, AnyStr]=None
                        )->None:
    save_dir = load_dir if save_dir is None else save_dir
    img_names = [img_name for img_name in os.listdir(load_dir) if '.ipynb' not in img_name]
    for img_name in img_names:
        make_binary_mask(os.path.join(save_dir, img_name))

def make_binary_mask(path:Union[Path, AnyStr]) -> None:
    path_new = path.split('.')[0] + '.tiff'
    print(path_new)
    img = np.array(Image.open(path))
    img_new = Image.fromarray(np.where(img>0,1.0,0.0).astype(np.float64))
    img_new.save(path_new)
    print(f'save the binary mask to {path_new}')  

def covert_annot_to_binary(filepath, save_to):
    if not os.path.isdir(save_to): os.mkdir(save_to)
    img = Image.open(filepath)
    img = np.asarray(img)
    img = np.where(img>0,1,0).astype(np.float64)
    Image.fromarray(img).save(os.path.join(save_to, os.path.split(filepath)[-1].split('.')[-2]+'.tiff'))
    print(f'conversion completed for {filepath}.')
    
def covert_annot_to_binary_list(file_folder, save_to, img_format:str='png'):
    root = os.path.split(file_folder)
    for f in sorted(os.listdir(file_folder)):
        if f.endswith(f'{img_format}'):
            covert_annot_to_binary(os.path.join(file_folder, f), save_to)
                      