import os
import numpy as np
from PIL import Image
from skimage.transform import resize
import matplotlib.pyplot as plt
from skimage import data, segmentation, feature, future
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from functools import partial
from joblib import dump, load
from typing import Dict, Optional, Union
from rich.progress import track
from .utils import get_file_paths, images_to_array, iou, calculate_iou_with_truth


class RandomForestSegmentor():
    def __init__(self, 
                 train_image_folder_path, 
                 train_annot_folder_path,
                 sigma_min=1,
                 sigma_max=16,
                 resize_factor:int=4,
                 model_kwargs=dict(n_estimators=100,
                                   max_depth=10,
                                   max_samples=0.25,
                                   n_jobs=-1),
                ):
        
        train_image_paths = get_file_paths(train_image_folder_path)
        train_annot_paths = get_file_paths(train_annot_folder_path)
        
        self.train_image = images_to_array(train_image_paths)
        self.train_annot = images_to_array(train_annot_paths)
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.resize_factor = resize_factor
        self.clf = RandomForestClassifier(**model_kwargs)
        
        assert (self.train_image[0].shape == self.train_annot[0].shape)
        self.train_image_size = [int(i/self.resize_factor) for i in self.train_image[0].shape[:2]]
        
        self.train_image_resized = [resize(img, (*self.train_image_size,1)) for img in self.train_image]
        self.train_annot_resized = [resize(img, (*self.train_image_size,1)) for img in self.train_annot]
        self.train_annot_resized = [(np.where(mask<0.5,0,1) + 1) for mask in self.train_annot_resized]
        self.num_pixels = self.train_annot_resized[0].shape[0] * self.train_annot_resized[0].shape[1]
        
        self.features_func = partial(feature.multiscale_basic_features,
                                     intensity=True, edges=True, texture=True,
                                     sigma_min=self.sigma_min, sigma_max=self.sigma_max,
                                     channel_axis=-1)
        
        self.features = np.concatenate([self.features_func(img) for img in self.train_image_resized],axis=1)
        self.labels = np.concatenate(self.train_annot_resized,axis=1).squeeze()
        
    def fit(self):
        future.fit_segmenter(self.labels, self.features, self.clf)
    
    def predict(self, image):
        image = resize(image, [int(i/self.resize_factor) for i in image.shape[:2]])
        features = self.features_func(image)
        return future.predict_segmenter(features, self.clf)
    
    def predict_folder_images(self,folder_path, save_path):
        if not os.path.isdir(save_path): os.mkdir(save_path)
        img_files = get_file_paths(folder_path)
        for img_file in track(img_files, description="Segmenting..."):
            img = np.asarray(Image.open(img_file))[:,:,np.newaxis]
            shape = img.squeeze().shape
            mask = self.predict(img)
            mask = resize(mask, shape)
            mask = np.where(mask<mask.mean(),0.0,1.0)
            mask = Image.fromarray(mask).save(os.path.join(save_path, img_file.split('/')[-1]))
            
    def save_model(self,name):
        dump(self.clf, f'{name}.joblib')
        
    def load_model(self,path):
        self.clf = load(path)

        
        

