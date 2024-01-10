import torch
from torch.optim.lr_scheduler import OneCycleLR


def create_optimizers(model, config):
    lr = config["lr"]
    max_epochs = config["epochs"]
    batch_size = config["batch_size"]

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=0.9,
        weight_decay=5e-4,
    )

    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    steps_per_epoch = 45000 // batch_size
    scheduler_dict = {
        "scheduler": OneCycleLR(
            optimizer,
            0.1,
            epochs=max_epochs,
            steps_per_epoch=steps_per_epoch,
        ),
        "interval": "step",
    }
    return {"optimizer": optimizer, "lr_scheduler": scheduler_dict}
