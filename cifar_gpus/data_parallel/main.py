"""Train CIFAR10 with PyTorch."""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

from tqdm import tqdm
from cifar_gpus.utils.arguments import arg_parse
from cifar_gpus.utils.datasets import create_datasets
from cifar_gpus.utils.model import create_model
from cifar_gpus.utils.optimizer import create_optimizers

device = "cuda" if torch.cuda.is_available() else "cpu"
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

args = arg_parse()

# Data
print("==> Preparing data..")

trainloader, testloader, _ = create_datasets(args.__dict__)

classes = (
    "plane",
    "car",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)

# Model
print("==> Building model..")
net = create_model()
net = net.to(device)
if device == "cuda":
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

criterion = F.nll_loss
optimizers = create_optimizers(net, args.__dict__)
optimizer = optimizers["optimizer"]
scheduler = optimizers["lr_scheduler"]


# Training
def train(epoch):
    print("\nEpoch: %d" % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    with tqdm(trainloader, unit="brach") as tepoch:
        for batch_idx, (inputs, targets) in enumerate(tepoch):
            tepoch.set_description(f"Train {epoch}")
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            tepoch.set_postfix(
                loss=train_loss / (batch_idx + 1),
                accuracy="Acc: %.3f%% (%d/%d)"
                % (100.0 * correct / total, correct, total),
            )


def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        with tqdm(testloader, unit="brach") as tepoch:
            for batch_idx, (inputs, targets) in enumerate(tepoch):
                tepoch.set_description(f"Test {epoch}")
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = net(inputs)
                loss = criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                tepoch.set_postfix(
                    loss=test_loss / (batch_idx + 1),
                    accuracy="Acc: %.3f%% (%d/%d)"
                    % (100.0 * correct / total, correct, total),
                )


for epoch in range(args.epochs):
    train(epoch)
    test(epoch)
    scheduler.step()
