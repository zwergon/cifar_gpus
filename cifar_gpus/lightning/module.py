import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchmetrics.functional import accuracy

from cifar_gpus.utils.model import create_model
from cifar_gpus.utils.optimizer import create_optimizers


class LitResnet(L.LightningModule):
    def __init__(self, batch_size=256, lr=0.05):
        super().__init__()

        self.save_hyperparameters()
        self.model = create_model()

    def forward(self, x):
        out = self.model(x)
        return F.log_softmax(out, dim=1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        self.log("train_loss", loss, sync_dist=True)
        return loss

    def evaluate(self, batch, stage=None):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y, task="multiclass", num_classes=10)

        if stage:
            self.log(f"{stage}_loss", loss, prog_bar=True, sync_dist=True)
            self.log(f"{stage}_acc", acc, prog_bar=True, sync_dist=True)

    def validation_step(self, batch, batch_idx):
        self.evaluate(batch, "val")

    def test_step(self, batch, batch_idx):
        self.evaluate(batch, "test")

    def configure_optimizers(self):
        config = {
            "lr": self.hparams.lr,
            "batch_size": self.hparams.batch_size,
            "epochs": self.trainer.max_epochs,
        }

        return create_optimizers(self.model, config)
