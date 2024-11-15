# packages
import logging
import time
from typing import *

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def evaluate(
    loader: DataLoader, model: nn.Module, cost: nn.Module, gpu: int = 0,  # evaluate
    logger: logging.Logger = None, title: str = ''  # logging
) -> Tuple[float, float]:
    """
    Evaluate model on a dataset
    Compute time, loss, and accuracy of evaluation

    Args:
        loader (DataLoader): data loader for evaluation
        model (nn.Module): model to evaluate
        cost (nn.Module): loss function
        gpu (int): gpu to use for computation
        logger (logging.Logger): logger for recording evaluation results
        title (str): title for logging

    Returns:
        Tuple[float, float]: evaluation loss and accuracy

    """

    # initializations
    model = model.eval()
    model = model.cuda(gpu)
    eval_loss = eval_acc = eval_n = 0

    # testing
    eval_start = time.time()
    for batch, (images, labels) in enumerate(loader):
        images, labels = images.cuda(gpu), labels.cuda(gpu)

        # forward
        with torch.no_grad():
            out = model(images)
            loss = cost(out, labels)

        # results
        _, preds = out.max(dim=1)
        eval_loss += loss.item() * labels.size(0)
        eval_acc += (preds == labels).sum().item()
        eval_n += labels.size(0)

    # summarize
    eval_loss /= eval_n
    eval_acc /= eval_n
    eval_end = time.time()

    if logger is not None:
        logger.info(
            title.upper() + ' - Time %.1f, Loss %.4f, Acc %.4f',
            eval_end - eval_start, eval_loss, eval_acc
        )

    return eval_loss, eval_acc


# evaluation that saves output distribution layer and labels
def evaluate_output(
    loader: DataLoader, model: nn.Module, cost: nn.Module, gpu: int = 0,  # evaluate
    logger: logging.Logger = None, title: str = '',  # logging
    output: bool = True,
):
    """
    Evaluate model on a dataset
    Compute time, loss, and accuracy of evaluation
    Save output distribution layer and labels

    Args:
        loader (DataLoader): data loader for evaluation
        model (nn.Module): model to evaluate
        cost (nn.Module): loss function
        gpu (int): gpu to use for computation
        logger (logging.Logger): logger for recording evaluation results
        title (str): title for logging
        output (bool): whether to save output distribution layer and labels

    Returns: Tuple[float, float, Union[None, np.ndarray], Union[None, np.ndarray]]: evaluation loss, accuracy,
        output distribution layer, and labels
    """

    # initializations
    model = model.eval()
    model = model.cuda(gpu)
    eval_loss = eval_acc = eval_n = 0
    if output:
        output_layers, output_labels = [], []
    else:
        output_layers, output_labels = None, None

    # testing
    eval_start = time.time()
    for batch, (images, labels) in enumerate(loader):
        images, labels = images.cuda(gpu), labels.cuda(gpu)

        # forward
        with torch.no_grad():
            out = model(images)
            loss = cost(out, labels)

        if output:
            output_layers.append(out)
            output_labels.append(labels)

        # results
        _, preds = out.max(dim=1)
        eval_loss += loss.item() * labels.size(0)
        eval_acc += (preds == labels).sum().item()
        eval_n += labels.size(0)

    # summarize
    eval_loss /= eval_n
    eval_acc /= eval_n
    eval_end = time.time()

    if logger is not None:
        logger.info(
            title.upper() + ' - Time %.1f, Loss %.4f, Acc %.4f',
            eval_end - eval_start, eval_loss, eval_acc
        )

    # collect output
    if output:
        output_layers = torch.cat(output_layers, dim=0).cpu().numpy()
        output_labels = torch.cat(output_labels, dim=0).cpu().numpy()

    return eval_loss, eval_acc, output_layers, output_labels
