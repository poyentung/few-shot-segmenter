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
    
    