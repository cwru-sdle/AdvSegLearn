import torch
import sys
import os
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))

from train_GAN import train_GAN, metrics_list,set_generator_optimizer,set_discriminator_optimizer,segmented_training_epoch,validation_epoch,unsegmented_training_epoch,save_results
from discriminator_loss import discriminator_loss
from semi_supervised_loss import semi_supervised_loss
from unet import Unet
data = torch.randn(10, 1, 28, 28)
targets = torch.randn(10, 1, 28, 28)
dataset = TensorDataset(data, targets)
seg_train_dl = DataLoader(dataset, batch_size=2)
seg_val_dl = DataLoader(dataset, batch_size=2)
unseg_train_dl = DataLoader(data, batch_size=2)

train_GAN(
    seg_disc_loss=discriminator_loss(),
    seg_gen_loss=torch.nn.BCELoss(),
    unseg_gen_loss=semi_supervised_loss(),
    seg_model=Unet(input_channels=1,output_channels=2),
    disc_model=Unet(input_channels=2,output_channels=1),
    seg_train_dl=seg_train_dl,
    seg_val_dl=seg_val_dl,
    unseg_train_dl = unseg_train_dl,
    epochs=2,
    accumulation_steps=5,
)
print('finished script')


def validate_epoch(model, val_dl, loss_fn):
    valid_losses = []
    for idx, datapoint in enumerate(val_dl):
        model_output = model(datapoint[0])
        loss = loss_fn(model_output, datapoint[1])
        valid_losses.append(loss.item())
    return valid_losses

def validation_epoch(seg_model, seg_val_dl, seg_gen_loss,val_gen_loss_metrics):
    val_gen_loss_temp = []
    for idx, datapoint in enumerate(seg_val_dl):
        model_output = seg_model(datapoint[0])
        #Save the validation loss 
        val_loss = seg_gen_loss(model_output,datapoint[1])
        val_gen_loss_temp.append(val_loss.item())
    val_gen_loss_metrics.append(metrics_list(val_gen_loss_temp))