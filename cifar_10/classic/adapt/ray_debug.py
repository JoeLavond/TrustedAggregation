""" Packages """
# base
import argparse
import copy
import os
from pathlib import Path
import sys
import time

# numeric
import numpy as np
import pandas as pd

# torch
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms as T
from torch.utils.data import Dataset, DataLoader

# tuning
from ray import tune
from ray.air import session, FailureConfig
from ray.air.config import RunConfig
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from ray.tune.stopper import TrialPlateauStopper

# source
sys.path.insert(2, f'{Path.home()}/')
import global_utils as gu

sys.path.insert(3, f'{Path.home()}/fed-tag/')
import proj_utils as pu

sys.path.insert(4, f'{Path.home()}/models/')
import resnet
import vgg


""" Setup """
# file control
def get_args():
    parser = argparse.ArgumentParser()

    """ File setup """
    # control
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--n_classes', default=10, type=int)
    # resources
    parser.add_argument('--cpus_per_trial', default=2, type=int)
    parser.add_argument('--gpus_per_trial', default=0.25, type=float)
    # output
    parser.add_argument('--resnet', default=1, type=int)
    parser.add_argument('--print_all', default=0, type=int)

    """ Federated learning """
    # basic fl
    parser.add_argument('--n_users', default=1, type=int)
    parser.add_argument('--n_user_data', default=500, type=int)
    parser.add_argument('--alpha', default=10000, type=int)
    # users
    parser.add_argument('--n_batch', default=64, type=int)
    parser.add_argument('--mom', default=0.9, type=float)
    parser.add_argument('--wd', default=5e-4, type=float)
    parser.add_argument('--n_epochs_pois', default=50, type=int)
    parser.add_argument('--lr_pois', default=0.01, type=float)
    parser.add_argument('--n_epochs', default=10, type=int)
    parser.add_argument('--lr', default=0.01, type=float)

    """ Data poisoning """
    # attack
    parser.add_argument('--dba', default=0, type=int)
    parser.add_argument('--p_pois', default=0.1, type=float)
    parser.add_argument('--target', default=0, type=int)
    parser.add_argument('--row_size', default=4, type=int)
    parser.add_argument('--col_size', default=4, type=int)
    # defense
    parser.add_argument('--d_scale', default=2, type=float)
    parser.add_argument('--d_smooth', default=1, type=int)
    parser.add_argument('--n_val_data', default=500, type=int)
    parser.add_argument('--alpha_val', default=10000, type=int)
    parser.add_argument('--remove_val', default=1, type=int)

    return parser.parse_args()


""" Main function HERE """

def main():

    # setup
    args = get_args()
    args.gpu_start = 0
    args.mu = .01

    # source
    sys.path.insert(2, f'{Path.home()}/')
    import global_utils as gu

    sys.path.insert(3, f'{Path.home()}/fed-tag/')
    import proj_utils as pu

    sys.path.insert(4, f'{Path.home()}/models/')
    import resnet
    import vgg

    gu.set_seeds(args.seed)

    """ Import data """
    cifar_trans = T.Compose([
        T.Pad(padding=4),
        T.RandomHorizontalFlip(),
        T.RandomCrop(size=32)
    ])

    train_data = datasets.CIFAR10(
        root=f'{Path.home()}/data/',
        train=True,
        download=True
    )

    train_data = pu.Custom3dDataset(train_data.data, train_data.targets, cifar_trans)
    cifar_mean = train_data.mean()
    cifar_std = train_data.std()

    """ Backdoor model """
    stamp_model = pu.BasicStamp(
        1, args.dba,
        row_size=args.row_size, col_size=args.col_size
    ).cuda(args.gpu_start)
    stamp_model = stamp_model.eval()

    # get user data
    [val_data_indices, user_data_indices] = train_data.sample(
        2, args.n_user_data, args.alpha, args.n_classes
    )

    # create loaders
    val_loader = DataLoader(
        train_data.get_user_data(
            val_data_indices, 0, 0, stamp_model,
            **vars(args)
        ),
        batch_size=args.n_batch,
        shuffle=True,
        num_workers=1,
        pin_memory=True
    )

    clean_loader = DataLoader(
        train_data.get_user_data(
            user_data_indices, 0, 1, stamp_model,
            **vars(args)
        ),
        batch_size=args.n_batch,
        shuffle=True,
        num_workers=1,
        pin_memory=True
    )

    pois_loader = DataLoader(
        train_data.get_user_data(
            user_data_indices, 1, 1, stamp_model,
            **vars(args)
        ),
        batch_size=args.n_batch,
        shuffle=True,
        num_workers=1,
        pin_memory=True
    )


    """ Clean model """
    cost = nn.CrossEntropyLoss()
    clean_model = nn.Sequential(
        gu.StdChannels(cifar_mean, cifar_std),
        (
            resnet.resnet18(num_classes=args.n_classes, pretrained=False)
            if args.resnet else vgg.vgg16_bn()
        )
    ).cpu()
    clean_model = clean_model.eval()

    # copy model
    pois_model = copy.deepcopy(clean_model).cpu()
    val_model = copy.deepcopy(clean_model).cpu()

    clean_opt = optim.SGD(
        clean_model.parameters(),
        lr=args.lr, weight_decay=args.wd, momentum=args.wd
    )

    clean_model = clean_model.cuda(args.gpu_start)
    _ = gu.training(
        clean_loader, clean_model, cost, clean_opt,
        args.n_epochs, args.gpu_start
    )
    clean_model = clean_model.cpu()


    """ Validation model """
    val_opt = optim.SGD(
        val_model.parameters(),
        lr=args.lr, weight_decay=args.wd, momentum=args.wd
    )

    val_model = val_model.cuda(args.gpu_start)
    _ = gu.training(
        val_loader, val_model, cost, val_opt,
        args.n_epochs, args.gpu_start
    )
    val_model = val_model.cpu()


    """ Poisoned model """
    pois_model = pois_model.train()
    pois_opt = optim.SGD(
        pois_model.parameters(),
        lr=args.lr_pois, weight_decay=args.wd, momentum=args.wd
    )

    for epoch in range(args.n_epochs_pois):

        """ Training """
        pois_model = pois_model.train()
        for batch, (images, labels) in enumerate(pois_loader):

            # initializations
            images, labels = images.cuda(args.gpu_start), labels.cuda(args.gpu_start)

            # forward
            pois_model = pois_model.cpu()
            with torch.no_grad():
                clean_model = clean_model.cuda(args.gpu_start)
                clean_out = clean_model(images)
                clean_model = clean_model.cpu()

            pois_model = pois_model.cuda(args.gpu_start)
            pois_out = pois_model(images)

            loss = cost(pois_out, labels)
            penalty = ((args.mu / 2) * torch.square(torch.norm(pois_out - clean_out)))

            # backward
            all_loss = loss + penalty
            pois_opt.zero_grad(set_to_none=True)
            all_loss.backward()
            pois_opt.step()


        """ Testing """
        pois_model = pois_model.eval()
        total_penalty = total_n = 0

        val_total_loss = val_total_acc = 0
        clean_total_loss = clean_total_acc = 0
        pois_total_loss = pois_total_acc = 0

        for batch, (images, labels) in enumerate(val_loader):

            # initializations
            images, labels = images.cuda(args.gpu_start), labels.cuda(args.gpu_start)

            # forward
            with torch.no_grad():

                pois_model = pois_model.cpu()

                clean_model = clean_model.cuda(args.gpu_start)
                clean_out = clean_model(images)
                clean_model = clean_model.cpu()

                val_model = val_model.cuda(args.gpu_start)
                val_out = val_model(images)
                val_model = val_model.cpu()

                pois_model = pois_model.cuda(args.gpu_start)
                pois_out = pois_model(images)

            penalty = torch.square(torch.norm(pois_out - val_out))
            total_penalty += penalty.item()

            # results
            total_n += labels.size(0)

            val_loss = cost(val_out, labels)
            _, val_preds = val_out.max(dim=1)
            val_total_loss += val_loss.item() * labels.size(0)
            val_total_acc += (val_preds == labels).sum().item()

            clean_loss = cost(clean_out, labels)
            _, clean_preds = clean_out.max(dim=1)
            clean_total_loss += clean_loss.item() * labels.size(0)
            clean_total_acc += (clean_preds == labels).sum().item()

            pois_loss = cost(pois_out, labels)
            _, pois_preds = pois_out.max(dim=1)
            pois_total_loss += pois_loss.item() * labels.size(0)
            pois_total_acc += (pois_preds == labels).sum().item()

        # summary
        val_total_loss /= total_n
        val_total_acc /= total_n

        clean_total_loss /= total_n
        clean_total_acc /= total_n

        pois_total_loss /= total_n
        pois_total_acc /= total_n

        print()
        print(f'Validation - Testing - Epoch: {epoch}, Loss: {val_total_loss:.4f}, Acc: {val_total_acc:.4f}')
        print(f'Clean - Testing - Epoch: {epoch}, Loss: {clean_total_loss:.4f}, Acc: {clean_total_acc:.4f}')
        print(f'Poisoned - Testing - Epoch: {epoch}, Loss: {pois_total_loss:.4f}, Acc: {pois_total_acc:.4f}')


if __name__ == "__main__":
    main()
