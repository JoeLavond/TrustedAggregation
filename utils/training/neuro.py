# packages
import logging
import re
import time
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader


# Base class
# implement Neurotoxin attack using model weights instead of gradients
class Neurotoxin:
    """
    Function:
        Compute model weight changes
        Find locations of p*100% of the largest values
        Create mask to identify the largest locations
        Given update, replace the largest locations with original values
    Usage:
        Modification of Neurotoxin attack based on weights
    Remarks:
        NEED TO COPY.DEEPCOPY(MODEL) AS INPUT TO MODEL
        EXCEPT WHEN APPLYING THE MASK WITH MASK_()
    """

    def __init__(
        self,
        model: nn.Module,
        p: float
    ):
        """
        Initialize Neurotoxin attack
        Create mask from random initialized model
        Store original model weights
        Store percentage of weights to mask

        Args:
            model (nn.Module): model to be attacked
            p (float): percentage of weights to mask

        """
        # initializations
        self.mask = None
        self.p = p

        self.old_model = model.cpu()
        with torch.no_grad():
            for (_, weight) in self.old_model.state_dict().items():
                weight.zero_()

        # create mask from random initialized model
        self.update_mask_(model)

    def update_mask_(self, new_model: nn.Module):
        """
        Create a flattened mask of model weights
        Omit positions of the largest changes in model weights

        Perform Neurotoxin attack
        Project attack onto weights not regularly updated
        Intended to produce lasting attacks

        Args:
            new_model (nn.Module): model to be attacked

        """

        # initializations
        weight_diffs = []
        weight_count = 0

        with torch.no_grad():
            for (name, new_weight), (_, old_weight) in zip(
                    new_model.state_dict().items(),
                    self.old_model.state_dict().items()
            ):

                # skip non-float type weights (ex. batch-norm)
                # store zero tensor instead
                if bool(re.search('^0.', name)) or not torch.is_floating_point(old_weight):
                    diff = torch.flatten(
                        torch.zeros_like(old_weight)
                    )

                # otherwise compute change in weights
                else:
                    diff = torch.flatten(
                        new_weight.cpu() - old_weight
                    )
                    diff = diff.abs()  # absolute change

                    # keep track of total number of float model weights
                    weight_count += len(diff)

                # append flattened weights
                weight_diffs.append(diff)

            # find k weights to mask from p
            k = int(self.p * weight_count)

            # concatenate weights to find top k
            weight_diffs = torch.cat(weight_diffs)
            (_, top_indices) = torch.topk(weight_diffs, k)

            # create mask from top k indices
            self.mask = torch.zeros_like(weight_diffs, dtype=torch.int64)
            self.mask[top_indices] = 1  # replace top changes

            # move new model to old model
            self.old_model = new_model.cpu()

    def mask_(self, model: nn.Module):
        """
        Use mask on proposed model update
        Replace locations of the largest change
        Use instead values from original model

        Apply after every model update when using Neurotoxin

        Args:
            model (nn.Module): model to be attacked

        """
        # initializations
        start_index = 0

        with torch.no_grad():
            for (_, new_weight), (_, old_weight) in zip(
                    model.state_dict().items(),
                    self.old_model.state_dict().items()
            ):

                # store dimensions to unflatten
                input_size = new_weight.size()
                input_device = new_weight.get_device()

                # flatten model weights
                new_weight_flat = torch.flatten(new_weight).cpu()
                flat_size = len(new_weight_flat)

                # if no weights need to be replaced, skip iter
                temp_mask = self.mask[start_index:flat_size]
                if temp_mask.sum().item() == 0:
                    continue

                # reset new weights back to global values where masked
                old_weight_flat = torch.flatten(old_weight)
                new_weight_flat[temp_mask] = old_weight_flat[temp_mask]

                # unflatten masked weights and reassign in-place
                new_weight_unflat = new_weight_flat.view(input_size)
                new_weight.copy_(
                    new_weight_unflat.cuda(input_device)
                )

                # update start_index
                start_index += flat_size


""" Training """


def nt_training(
        loader: DataLoader, model: nn.Module, cost: nn.Module, opt: Optimizer, n_epochs: int = 1, gpu: int = 0, scheduler=None,  # training
        logger: logging.Logger = None, title: str = 'training', print_all: bool = False,  # logging
        nt_obj: Neurotoxin = None  # add neurotoxin attack
):
    """
    Train model on a dataset
    Compute time, loss, and accuracy of training

    Use the Neurotoxin object to apply Neurotoxin attack
    Project attack onto weights not regularly updated

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
        nt_obj (Neurotoxin): Neurotoxin object to apply attack

    Returns:
        Tuple[float, float]: training loss and accuracy

    """
    # initializations
    model = model.train()
    model = model.cuda(gpu)
    train_loss = train_acc = 0

    for epoch in range(n_epochs):
        epoch += 1

        # re-initializations
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
            if nt_obj is not None:
                nt_obj.mask_(model)

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
