import os
import hydra
import numpy as np
from omegaconf import DictConfig
import pytorch_lightning as pl
from typing import List, Optional
from sav.utils import hydra_logging
from sav.module.fs_segmenter import FewShotSegmenter
from sav.utils.annotator import Annotator
from pytorch_lightning import LightningModule

log = hydra_logging.get_logger(__name__)

def segment(config: DictConfig):
    # create save directory
    if not os.path.exists(config.save_dir):
        os.makedirs(config.save_dir)

    # Init lightning model
    log.info(f"Instantiating model <{config.model._target_}>")
    model: FewShotSegmenter = hydra.utils.instantiate(config.model).load_from_checkpoint(config.ckpt_path)
    
    # Init annotator
    log.info(f"Instantiating annotator <{config.annotator._target_}>")
    annotator: Annotator = hydra.utils.instantiate(config.annotator, model=model)
    
    #Segmenting images
    annotator(query_img_path = config.query_img_path,
              support_imgs_dir = config.support_imgs_dir,
              support_annots_dir = config.support_annots_dir,
              save_dir = config.output_dir)
    
    # Benchmarking
    if config.truth_annots_dir is not None:
        pred_annots_path = os.path.join(config.save_dir, 'outputs/mask')
        array = annotator.benchmark_iou(save_csv_name=config.name+'.csv',
                                        pred_annots_path=pred_annots_path,
                                        truth_annots_path=config.truth_annots_dir, 
                                        img_indices=(150,-170,150,-170))
        
        array = array[np.nonzero(array[:1000])]
        log.info(f'Mean: {array.mean():.4f}, Std: {array.std():.4f}')
    
    # Make sure everything closed properly
    log.info("Completed!")

    # Print path that saves the segmented images
    log.info(f"Segmented images saved to {config.save_dir}")

@hydra.main(version_base="1.2", config_path="conf", config_name="segment.yaml")
def main(config: DictConfig):
    
    # Applies optional utilities
    hydra_logging.extras(config)

    # Train model
    segment(config)

if __name__ == "__main__":
    main()