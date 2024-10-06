from td.model.model import MLP
import torch.nn as nn
import lightning as L
import torch
import numpy as np
from td.model.model import TrackGenerator
from torch.nn import MSELoss, BCELoss

class MLPL(L.LightningModule):
    def __init__(self):
        super(MLPL, self).__init__()
        self.model = MLP()

    def forward(self, x):
        x = self.model(x)
        return x

    def training_step(self, batch , batch_idx):

        x, y = batch
   
        y_hat = self.model(x)

        rmse = self.masked_rmse(y_hat, y)
        dist_loss = self.distance_between_points(y_hat)

        loss = rmse +  0.1 *dist_loss

        self.log("train/loss", loss, prog_bar=True)
        self.log("train/dist_loss", dist_loss, prog_bar=True)
        self.log("train/rmse", rmse, prog_bar=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        x,y = batch

        y_hat = self.model(x)
        loss = self.masked_rmse(y_hat, y)
        self.log("val/loss", loss, prog_bar=True)
        return loss
    
    def configure_optimizers(self):
        # optimizer and scheduler

        optimizer = torch.optim.Adam(self.parameters(), lr=0.00005)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)

        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val/loss"}
    
    def masked_rmse(self, y_pred, y_true,  mask_value=-99):
        # Create a mask where true values are not equal to the mask value
        mask = (y_true > mask_value)
        
        # Apply the mask to both y_true and y_pred
        y_true_masked = y_true[mask]
        y_pred_masked = y_pred[mask]

        # Calculate the RMSE
        mse = torch.mean((y_true_masked - y_pred_masked) ** 2)
        rmse = torch.sqrt(mse)
        
        return rmse

    def distance_between_points(self, y_hat):
        d = torch.diff(y_hat, dim=-1)
        dist = torch.hypot(d[:,0,:], d[:,1,:])

        return dist.mean()
    

class TrackGeneratorL(L.LightningModule):
    def __init__(self):
        super(TrackGeneratorL, self).__init__()
        self.model = TrackGenerator()

    def forward(self, x):
        sequence, logits = self.model(x)
        return sequence, logits

    def batch_transform(self, batch):
        x, y = batch
        y = y.permute(0,2,1)
        return x, y
    def training_step(self, batch , batch_idx):

        x, y = self.batch_transform(batch)
   
        sequence, log_probits = self.model(x)

        rmse = self.masked_rmse(sequence, y)
        bce = self.masked_bce(log_probits, y)

        loss = rmse +  bce

        self.log("train/loss", loss, prog_bar=True)
        self.log("train/bce", bce, prog_bar=True)
        self.log("train/rmse", rmse, prog_bar=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        x,y = self.batch_transform(batch)

        sequence, log_probits = self.model(x)
        rmse = self.masked_rmse(sequence, y)
        bce = self.masked_bce(log_probits, y)
        loss = rmse + bce

        self.log("val/loss", loss, prog_bar=True)
        self.log("val/bce", bce, prog_bar=True)
        self.log("val/rmse", rmse, prog_bar=True)
        return loss
    
    def configure_optimizers(self):
        # optimizer and scheduler

        optimizer = torch.optim.Adam(self.parameters(), lr=0.00005)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)

        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val/loss"}
    
    def masked_rmse(self, y_pred, y_true):
        # Create a mask where true values are not equal to the mask value
        mask = ~torch.isnan(y_true)
        
        # Apply the mask to both y_true and y_pred
        y_true_masked = y_true[mask]
        y_pred_masked = y_pred[mask]

        # Calculate the RMSE
        mse = torch.mean((y_true_masked - y_pred_masked) ** 2)
        rmse = torch.sqrt(mse)
        
        return rmse
    
    
    def masked_bce(self, y_pred, y_true):

        # Find seequence length based on first non nan values
        seq_len =  torch.argmin(torch.nan_to_num(y_true[:, :, 0], -1), dim=1)
        # Create a one hot encoding for sequence length
        y_true = torch.nn.functional.one_hot(seq_len, num_classes=200).to(torch.float32)

        # Calculate the binary cross entropy
        bce = BCELoss()(y_pred, y_true)
        return bce

    