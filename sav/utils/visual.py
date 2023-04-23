import numpy as np
import torch
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as patches
import matplotlib.colors as mcolors
from matplotlib.figure import Figure 
from pathlib import Path
from typing import Union, AnyStr
from PIL import Image

def log_fig(log_name:str, 
            fig:Figure, 
            logger:Union[pl.loggers.TensorBoardLogger, pl.loggers.WandbLogger], 
            current_epoch):
    
    fig.canvas.draw()

    # grab the pixel buffer and dump it into a numpy array
    eval_result = np.array(fig.canvas.renderer.buffer_rgba())
    plt.close()

    # for tensorboard
    if type(logger) == pl.loggers.TensorBoardLogger:
        eval_result = np.moveaxis(eval_result[:,:,:3],2,0)
        logger.experiment.add_image(f'{log_name}_{current_epoch}', eval_result)
    elif type(logger) == pl.loggers.WandbLogger:
        eval_result = Image.fromarray(eval_result[:,:,:3])
        logger.log_image(key=f'{log_name}_{current_epoch}', images=[eval_result])

def plot_evaluation(batch, nrow=2, ncol=4, cmap_annot='cividis',cmap_annot_hat='cividis'):
    indices = np.random.choice(np.arange(batch['query_img'].size(0)), size=4)
    query_img = batch['query_img'][indices].squeeze().detach().cpu().numpy()
    query_annot = batch['query_annot'][indices].squeeze().detach().cpu().numpy()
    query_annot_hat = batch['query_annot_hat'][indices]
    query_annot_hat_binary = torch.where(query_annot_hat>0.5,1.0,0.0).squeeze().detach().cpu().numpy()
    
    nrow = 2 
    ncol = 4
    fig, axs = plt.subplots(nrow, ncol, figsize=(1.5*ncol,1.5*nrow), dpi=150)
    for i in range(nrow):
        for j in range(ncol):
            if i == 0:
                axs[i,j].imshow(query_img[j], cmap='gray')
                axs[i,j].imshow(query_annot[j], cmap=cmap_annot, alpha=0.3)
            else:
                axs[i,j].imshow(query_img[j], cmap='gray')
                axs[i,j].imshow(query_annot_hat_binary[j], cmap=cmap_annot_hat, alpha=0.4)

            axs[i,j].axis("off")

    fig.subplots_adjust(wspace=0.0, hspace=0.05)
    return fig

def eval_model(module:pl.LightningModule, 
               dataset:torch.utils.data.Dataset, 
               phase_dir:Union[Path, AnyStr],
               cmap_annot='cividis',
               cmap_annot_hat='cividis'):
    nrow = 2
    ncol = 4
    batch_list = []
    for i in range(ncol):
        batch_list.append(dataset.__getitem__(i, phase_dir))
    
    device = module.device
    batch_dict = {}
    batch_dict['query_img'] = torch.stack([batch['query_img'] for batch in batch_list])
    batch_dict['query_annot'] = torch.stack([batch['query_annot'] for batch in batch_list])
    batch_dict['support_imgs'] = torch.stack([batch['support_imgs'] for batch in batch_list])
    batch_dict['support_annots'] = torch.stack([batch['support_annots'] for batch in batch_list])
    
    with torch.no_grad():
        batch_dict['query_annot_hat'] = module(query_img=batch_dict['query_img'].to(device=device), 
                                               support_imgs=batch_dict['support_imgs'].to(device=device), 
                                               support_annots=batch_dict['support_annots'].to(device=device))
    return plot_evaluation(batch_dict, nrow, ncol, cmap_annot, cmap_annot_hat)