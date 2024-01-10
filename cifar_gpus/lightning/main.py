import lightning as L

from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.loggers import MLFlowLogger

from cifar_gpus.lightning.module import LitResnet
from cifar_gpus.utils.arguments import arg_parse
from cifar_gpus.utils.datasets import create_datasets

L.seed_everything(7)

args = arg_parse()

train_dataloader, val_dataloader, test_dataloader = create_datasets(args.__dict__)

model = LitResnet(lr=args.lr, batch_size=args.batch_size)

trainer = L.Trainer(
    max_epochs=args.epochs,
    accelerator="auto",
    devices=args.devices,
    logger=MLFlowLogger(),
    callbacks=[LearningRateMonitor(logging_interval="step")],
)

trainer.fit(model, train_dataloader, val_dataloaders=val_dataloader)
trainer.test(model, test_dataloader)
