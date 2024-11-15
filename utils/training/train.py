# packages
from typing import *
import logging
import time

from torch import nn
from torch.utils.data import DataLoader


def training(
    loader: DataLoader, model: nn.Module, cost: nn.Module, opt, n_epochs: int = 1, gpu: int = 0, scheduler=None,  # training
    logger: logging.Logger = None, title: str = 'training', print_all: bool = False  # logging
) -> Tuple[float, float]:
    """
    Train model on a dataset
    Compute time, loss, and accuracy of training

    Args:
        loader (DataLoader): data loader for training
        model (nn.Module): model to train
        cost (nn.Module): loss function
        opt (Optimizer): optimizer for training
        n_epochs (int): number of epochs to train
        gpu (int): gpu to use for computation
        scheduler (torch.optim.lr_scheduler): learning rate scheduler
        logger (logging.Logger): logger for recording training results
        title (str): title for logging
        print_all (bool): print all epochs

    """
    # initializations
    model = model.train()
    model = model.cuda(gpu)
    train_loss = train_acc = 0

    for epoch in range(n_epochs):
        epoch += 1

        # initializations
        model = model.train()
        train_loss = train_acc = train_n = 0

        # training
        train_start = time.time()
        for batch, (images, labels) in enumerate(loader):
            images, labels = images.cuda(gpu), labels.cuda(gpu)

            # forward
            out = model(images)
            loss = cost(out, labels)

            # backward
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if scheduler is not None:
                scheduler.step()

            # results
            _, preds = out.max(dim=1)
            train_loss += loss.item() * labels.size(0)
            train_acc += (preds == labels).sum().item()
            train_n += labels.size(0)

        # summarize
        train_end = time.time()
        train_loss /= train_n
        train_acc /= train_n

        if (logger is not None) and (print_all or (epoch == n_epochs)):
            logger.info(
                title.upper() + ' - Epoch: %d, Time %.1f, Loss %.4f, Acc %.4f',
                epoch, train_end - train_start, train_loss, train_acc
            )

    return train_loss, train_acc
