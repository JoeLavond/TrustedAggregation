# packages
import logging
import numpy as np
import os
import random
import time
import torch
import torch.nn as nn


""" Setup """
# format logging
def get_log(path='.', f='log_train.log'):

    logger = logging.getLogger()
    logging.basicConfig(
        format='[%(asctime)s] - %(message)s',  # time stamps
        datefmt='%Y/%m/%d %H:%M:%S',  # time fmt
        level=logging.DEBUG,  # display to console
        handlers=[  # overwrite old logsLq
            logging.FileHandler(os.path.join(path, f), 'w+'),
            logging.StreamHandler()
        ]
    )
    return logger


# reproducibility
def set_seeds(seed):

    # set seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False

    return None


""" Modeling """
# standardize
class StdChannels(nn.Module):

    def __init__(self, mu, sd):
        super(StdChannels, self).__init__()
        self.register_buffer('mu', mu.view(len(mu), 1, 1))
        self.register_buffer('sd', sd.view(len(sd), 1, 1))

    def forward(self, x):
        return (x - self.mu) / self.sd


""" Training """
def evaluate(
    loader, model, cost, gpu=0,  # evaluate
    logger=None, title=''  # logging
    ):

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

    return (eval_loss, eval_acc)


def training(
    loader, model, cost, opt, n_epochs=1, gpu=0, scheduler=None,  # training
    logger=None, title='training', print_all=0  # logging
    ):

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

        if ((logger is not None) and (print_all or (epoch == n_epochs))):
            logger.info(
                title.upper() + ' - Epoch: %d, Time %.1f, Loss %.4f, Acc %.4f',
                epoch, train_end - train_start, train_loss, train_acc
            )

    return (train_loss, train_acc)
