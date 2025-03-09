import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from torchvision import transforms
from torchvision.models import vgg, resnet
import pytorch_lightning as pl
from torchmetrics.classification import BinaryJaccardIndex
from sav.utils.visual import log_fig, plot_evaluation


class BaseModule(pl.LightningModule):
    def __init__(
        self,
        backbone: str = "vgg16",
        optimizer: str = "adam",
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
    ):
        super().__init__()

        self.backbone_type = backbone
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.criterion = nn.BCEWithLogitsLoss()
        self.calculate_IoU = BinaryJaccardIndex(threshold=0.5)

    def forward(self, query_img, support_imgs, support_annots):
        return NotImplementedError

    def training_step(self, train_batch, batch_idx):
        query_img = train_batch["query_img"]  # size = [1,1,512,512]
        query_annot = train_batch["query_annot"]  # size = [1,1,512,512]
        support_imgs = train_batch["support_imgs"]  # size = [nshot,1,512,512]
        support_annots = train_batch["support_annots"]  # size = [nshot,1,512,512]

        query_annot_hat = self(query_img, support_imgs, support_annots)

        loss = self.criterion(query_annot_hat, query_annot)
        iou = self.calculate_IoU(F.sigmoid(query_annot_hat), query_annot)

        metrics = {
            "loss": loss,
            "iou": iou,
        }

        self.log("train/loss", metrics["loss"], prog_bar=True, on_step=True)
        self.log("train/iou", metrics["iou"], prog_bar=True, on_step=True)

        return metrics

    def train_epoch_end(self, training_step_outputs):
        epoch_train_loss = torch.stack(
            [x["loss"] for x in training_step_outputs]
        ).mean()
        self.log("train/loss_epoch", epoch_train_loss, prog_bar=True, on_step=False)

    def validation_step(self, val_batch, batch_idx):
        query_img = val_batch["query_img"]  # size = [bs,1,512,512]
        query_annot = val_batch["query_annot"]  # size = [bs,1,512,512]
        support_imgs = val_batch["support_imgs"]  # size = [bs,nshot,1,512,512]
        support_annots = val_batch["support_annots"]  # size = [bs,nshot,1,512,512]
        specimen_phase_name = val_batch["specimen_phase_name"]

        query_annot_hat = self(query_img, support_imgs, support_annots)
        loss = self.criterion(query_annot_hat, query_annot)

        iou = self.calculate_IoU(query_annot_hat, query_annot)

        metrics = {
            "val_loss": loss,
            "val_iou": iou,
            "class": specimen_phase_name[0],
            "query_img": query_img,
            "query_annot": query_annot,
            "query_annot_hat": query_annot_hat,
        }

        self.log("val/val_loss", metrics["val_loss"], prog_bar=False, on_step=True)
        self.log("val/val_iou", metrics["val_iou"], prog_bar=False, on_step=True)

        return metrics

    def validation_epoch_end(self, valid_step_outputs):
        epoch_val_loss = torch.stack([x["val_loss"] for x in valid_step_outputs]).mean()
        epoch_val_iou = torch.stack([x["val_iou"] for x in valid_step_outputs]).mean()
        self.log("val/epoch_val_loss", epoch_val_loss, prog_bar=True, on_step=False)
        self.log("val/epoch_val_iou", epoch_val_iou, prog_bar=True, on_step=False)

        # calculate iou per class
        class_iou = {}
        for output in valid_step_outputs:
            if output["class"] not in class_iou.keys():
                class_name = output["class"]
                class_iou[class_name] = [output["val_iou"]]
            else:
                class_iou[class_name].append(output["val_iou"])

        for class_name in class_iou.keys():
            class_iou[class_name] = (
                torch.stack(class_iou[class_name]).mean().detach().cpu().numpy()
            )
            self.log(f"metric/mean_iou_{class_name}", float(class_iou[class_name]))

        metrics_list = list()
        for i, class_name in enumerate(class_iou.keys()):
            metrics_list.append([class_name, class_iou[class_name]])

        self.logger.log_table(
            key=f"metrics_per_class_{self.current_epoch}",
            columns=["class_name", "mean_iou"],
            data=metrics_list,
        )

        self.metrics = pd.DataFrame.from_dict(
            class_iou, orient="index", columns=["mean_iou"]
        )

        # plot validation results
        last_batch = valid_step_outputs[-1]
        fig = plot_evaluation(last_batch)

        log_fig(
            fig=fig,
            log_name="visual/eval_check",
            logger=self.logger,
            current_epoch=self.current_epoch,
        )

    def test_step(self, test_batch, batch_idx):
        return self.validation_step(test_batch, batch_idx)

    def test_epoch_end(self, test_step_outputs):
        self.on_validation_epoch_end(test_step_outputs)

    def configure_optimizers(self):
        if self.optimizer == "adam":
            optimizer = torch.optim.Adam(
                self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
            )
        elif self.optimizer == "sgd":
            optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate)
        else:
            raise Exception("Unknown optimizer. Only adam is implemented.")
        return optimizer

class DeepLabV3Module(pl.LightningModule):
    def __init__(self, 
                 backbone:str='deeplabv3_resnet50', 
                 optimizer:str='adam', 
                 learning_rate:float=1e-4, 
                 weight_decay:float=1e-5
                ):
        super().__init__()
        
        self.backbone_type = backbone
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.criterion = nn.BCEWithLogitsLoss()
        self.calculate_IoU = BinaryJaccardIndex(threshold=0.5)
    
    def forward(self, query_img, support_imgs, support_annots):
        return NotImplementedError
    
    def training_step(self, train_batch, batch_idx):
        query_img      = train_batch['query_img']        # size = [1,1,512,512]
        query_annot    = train_batch['query_annot']      # size = [1,1,512,512]
        query_annot_hat = self(query_img)

        loss = self.criterion(query_annot_hat, query_annot)
        iou = self.calculate_IoU(F.sigmoid(query_annot_hat), query_annot)

        metrics = {
            'loss':loss,
            'iou':iou,
        }

        self.log('train/loss', metrics['loss'], prog_bar=True, on_step=True)
        self.log('train/iou', metrics['iou'], prog_bar=True, on_step=True)

        return metrics

    def train_epoch_end(self, training_step_outputs):
        epoch_train_loss = torch.stack([x['loss'] for x in training_step_outputs]).mean()
        self.log('train/loss_epoch', epoch_train_loss, prog_bar=True, on_step=False)

    def validation_step(self, val_batch, batch_idx):
        query_img      = val_batch['query_img']        # size = [bs,1,512,512]
        query_annot    = val_batch['query_annot']      # size = [bs,1,512,512]
        specimen_phase_name = val_batch['specimen_phase_name'] 
        
        query_annot_hat = self(query_img)
        loss = self.criterion(query_annot_hat, query_annot)
        
        iou = self.calculate_IoU(query_annot_hat, query_annot)
        
        metrics = {
            'val_loss':loss,
            'val_iou': iou,
            'class': specimen_phase_name[0],
            'query_img':query_img,
            'query_annot':query_annot,
            'query_annot_hat':query_annot_hat,
        }

        self.log('val/val_loss', metrics['val_loss'], prog_bar=False, on_step=True)
        self.log('val/val_iou', metrics['val_iou'], prog_bar=False, on_step=True)

        return metrics
    
    def validation_epoch_end(self, valid_step_outputs):
        epoch_val_loss = torch.stack([x['val_loss'] for x in valid_step_outputs]).mean()
        epoch_val_iou = torch.stack([x['val_iou'] for x in valid_step_outputs]).mean()
        self.log('val/epoch_val_loss', epoch_val_loss, prog_bar=True, on_step=False)
        self.log('val/epoch_val_iou', epoch_val_iou, prog_bar=True, on_step=False)
        
        # calculate iou per class
        class_iou = {}
        for output in valid_step_outputs:
            if output['class'] not in class_iou.keys():
                class_name = output['class']
                class_iou[class_name] = [output['val_iou']]
            else:
                class_iou[class_name].append(output['val_iou'])
        
        for class_name in class_iou.keys():
            class_iou[class_name] = torch.stack(class_iou[class_name]).mean().detach().cpu().numpy()
            self.log(f'metric/mean_iou_{class_name}', float(class_iou[class_name]))
        
        metrics_list = list()
        for i, class_name in enumerate(class_iou.keys()):
            metrics_list.append([class_name, class_iou[class_name]])
        
        self.logger.log_table(key=f'metrics_per_class_{self.current_epoch}', 
                              columns=['class_name', 'mean_iou'], 
                              data=metrics_list)
        
        self.metrics = pd.DataFrame.from_dict(class_iou, orient='index', columns=['mean_iou'])

        # plot validation results
        last_batch = valid_step_outputs[-1]
        fig = plot_evaluation(last_batch)
        
        log_fig(fig=fig,
                log_name='visual/eval_check',  
                logger=self.logger, 
                current_epoch=self.current_epoch)
        
    def test_step(self, test_batch, batch_idx):
        return self.validation_step(test_batch, batch_idx)
    
    def test_epoch_end(self, test_step_outputs):
        self.on_validation_epoch_end(test_step_outputs)

    def configure_optimizers(self):
        if self.optimizer == "adam":
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay)
        else:
            raise Exception("Unknown optimizer. Only adam is implemented.")


def set_greyscale_weights(model: vgg.VGG) -> vgg.VGG:
    weights = model.features[0].weight.data
    kernel_out_channels, kernel_in_channels, kernel_rows, kernel_columns = weights.shape
    grayscale_weights = torch.zeros(
        (kernel_out_channels, 1, kernel_rows, kernel_columns)
    )
    grayscale_weights = (
        (weights[:, 0, :, :] * 0.2989)
        + (weights[:, 1, :, :] * 0.5870)
        + (weights[:, 2, :, :] * 0.1140)
    )

    # Set weights for first Conv2d layer t oadapt greyscal images
    model.features[0].weight.data = grayscale_weights.unsqueeze(1)
    return model


class TimeDistributed(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, x):
        if len(x.size()) <= 2:
            return self.module(x)
        n, t = x.size()[:2]
        x_reshape = x.view(n * t, *x.size()[2:])
        y = self.module(x_reshape)
        y = y.view(n, t, *y.size()[1:])
        return y

class GlobalAveragePooling2D(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, support_annots, support_imgs_encoded):
        # support_imgs_encoded size = [bs, nshot, c, w', h']
        # support_annots size = [bs, nshot, 1, w, h]
        nchannels = support_imgs_encoded.size(2)
        encoded_input_size = support_imgs_encoded.size()[-2:]  # get [w', h']

        # make the annot the size = [bs, nshot, 1, w', h']
        support_annots_resized = transforms.Resize(encoded_input_size)(
            support_annots.squeeze(2)
        ).unsqueeze(2)

        # make the annot the size = [bs, nshot, c, w', h']
        support_annots_resized = torch.tile(
            support_annots_resized, (1, 1, nchannels, 1, 1)
        )
        support_imgs_masked = torch.multiply(
            support_imgs_encoded, support_annots_resized
        )  # element-wise multiplication

        # AvgPool along [nshot,w',h'] -> size = [bs,1,c,1,1]
        support_imgs_masked = torch.div(
            support_imgs_masked.sum(dim=[1, 3, 4], keepdims=True),
            support_annots_resized.sum(dim=[1, 3, 4], keepdims=True),
        )
        if torch.isnan(support_imgs_masked).any():
            support_imgs_masked = torch.nan_to_num(support_imgs_masked)

        support_imgs_masked = torch.tile(
            support_imgs_masked.squeeze(1),
            (1, 1, encoded_input_size[0], encoded_input_size[1]),
        )  # size = [bs,c,w',h']

        return support_imgs_masked
