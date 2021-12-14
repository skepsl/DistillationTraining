import argparse

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision.datasets as datasets

from trainer import Trainer


def args():
    parser = argparse.ArgumentParser()

    hpstr = "Running Mode"
    parser.add_argument('--mode', default='train',
                        nargs='*', type=str, help=hpstr)

    hpstr = "Epoch training"
    parser.add_argument('--epochs', default=300,
                        nargs='*', type=int, help=hpstr)

    hpstr = "Batch size"
    parser.add_argument('--batch', default=128,
                        nargs='*', type=int, help=hpstr)

    hpstr = "Number of worker"
    parser.add_argument('--worker', default=4,
                        nargs='*', type=int, help=hpstr)

    hpstr = "Whether log to wandb"
    parser.add_argument('--wandb', default=False,
                        nargs='*', type=int, help=hpstr)

    args = parser.parse_args()
    return args


def run(args):
    transform_train = transforms.Compose([
        transforms.Resize(224,),
        transforms.RandomCrop(224, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
    ])

    transform_test = transforms.Compose([
        transforms.Resize(224, ),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
    ])

    trainset = datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batch, shuffle=True, num_workers=2)

    testset = datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)
    val_loader = torch.utils.data.DataLoader(
        testset, batch_size=args.batch, shuffle=False, num_workers=2)

    length = [50000, 10000]
    trainer = Trainer(args)
    trainer.train(train_loader, val_loader, length)


if __name__ == '__main__':
    run(args())
