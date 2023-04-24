import os
import hydra
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
    model: FewShotSegmenter = hydra.utils.instantiate(config.model)
    model.load_from_checkpoint(config.ckpt_path)
    
    # Init annotator
    log.info(f"Instantiating annotator <{config.annotator._target_}>")
    annotator: Annotator = hydra.utils.instantiate(config.annotator, model=model)
    
    # Segmenting images
    annotator(query_img_path = config.query_img_path,
              support_imgs_dir = config.support_imgs_dir,
              support_annots_dir = config.support_annots_dir,
              save_dir = config.output_dir)
    
    # Send some parameters from config to all lightning loggers
    log.info("Logging hyperparameters!")
    hydra_logging.log_hyperparameters(
        config=config,
        model=model,
        annotator=annotator
    )
        
    # Make sure everything closed properly
    log.info("Finalizing!")

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