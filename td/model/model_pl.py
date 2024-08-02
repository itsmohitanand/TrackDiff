from td.model.model import MLP
import torch.nn as nn
import lightning as L
import torch
import numpy as np

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
        loss = self.masked_rmse(y_hat, y)
        self.log("train/loss", loss, prog_bar=True)
        
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