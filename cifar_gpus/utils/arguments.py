import os
import torch
import argparse

PATH_DATASETS = os.environ.get("PATH_DATASETS", ".")
BATCH_SIZE = 256 if torch.cuda.is_available() else 64
NUM_WORKERS = int(os.cpu_count() / 2)


def arg_parse():
    parser = argparse.ArgumentParser(description="PyTorch CIFAR10 Training")
    parser.add_argument(
        "-p", "--path", default=PATH_DATASETS, type=str, help="path to dataset"
    )
    parser.add_argument(
        "-d", "--devices", default=1, type=int, help="number of devices"
    )
    parser.add_argument(
        "-b", "--batch_size", default=BATCH_SIZE, type=int, help="number of devices"
    )
    parser.add_argument(
        "-n", "--num_workers", default=NUM_WORKERS, type=int, help="number of devices"
    )
    parser.add_argument("-e", "--epochs", default=30, type=int, help="number of epochs")
    parser.add_argument("-l", "--lr", default=0.05, type=float, help="learing rate")

    return parser.parse_args()
