import os
import numpy as np
from pathlib import Path
from typing import Union, Dict, Tuple, AnyStr
from PIL import Image, ImageOps


def sampling_from_dir(path:Union[Path, AnyStr])->Union[Path, AnyStr]:
    """
    Ramdonly sample a file_name from a diretory, and return its full path of the file or the directory.
    """
    file_names = [file_name for file_name in os.listdir(path) if '.ipynb' not in file_name]
    fine_name = np.random.choice(file_names, 1, replace=False)[0]
    return os.path.join(path, fine_name)

def sample_paired_img_annot(phase_dir:Union[Path, AnyStr])->Tuple["numpy.ndarray", "numpy.ndarray"]:
    """
    Return a tuple conatining a pair of image and its annotation (binary mask) sampled from the phase_dir
    An example of the phase_dir:
    
    phase_dir/
       ├── annotation/        # target masks
       └── image/             # input images
    """
    # Randomly sample an img from the phase
    img_dir = os.path.join(phase_dir, 'image')
    img_path = sampling_from_dir(img_dir)
    img = np.array(Image.open(img_path)).astype(np.float64)

    # Get the corresponding annotation of the img
    annot_dir = os.path.join(phase_dir, 'annotation')
    annot_path = os.path.join(phase_dir, 'annotation', img_path.split('/')[-1])
    annot = np.array(Image.open(annot_path)).astype(np.float64)
    return (img, annot)

def get_img_sizes(datapath:Union[Path, AnyStr])-> Dict[str, Tuple[int,int]]:
    """
    Return a dictonary conatining tuples of image sizes of different phases, e.g., {'cauliflower':(2048,1536),}
                                                                                    'apollo_70017':(1004,1024)}
    """
    sample_names = [file_name for file_name in os.listdir(datapath) if '.' not in file_name] # extract file name under the datapath dir (e.g. dir/cauliflower)
    img_sizes = {}
    for sample_name in sample_names:
        phase_dir = sampling_from_dir(os.path.join(datapath, sample_name))
        dummy_img_path = sampling_from_dir(os.path.join(phase_dir, 'image'))
        img_sizes[sample_name] = Image.open(dummy_img_path).size
    return img_sizes

def downsample_and_pad(img: "PIL.Image", 
                       patch_size:Tuple[int], 
                       down_sampling:int
                      ) -> "PIL.Image":
    resized_img = img.resize((img.size[0]//down_sampling, img.size[1]//down_sampling))
    width, height = resized_img.size
    size_width = patch_size[0]*(width//patch_size[0]+1)
    size_height = patch_size[1]*(height//patch_size[1]+1)
    padded_img = ImageOps.pad(resized_img, size=(size_width, size_height))
    return padded_img