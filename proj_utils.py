# packages
import copy
import math
import numpy as np
import sys
import time
from functools import partial

import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.distributions.dirichlet import Dirichlet

# source
sys.path.insert(2, '/home/joe/')
import global_utils as gu


""" Data """
class Custom2dDataset(Dataset):

    """ Initial Setup """
    def __init__(self, images, labels, transformations=None, init=0):

        # load data
        self.transformations = transformations

        if not isinstance(images, torch.Tensor):
            self.images = torch.from_numpy(images).float()
            self.labels = torch.from_numpy(labels)
        else:
            self.labels = labels.clone().detach()
            self.images = images.clone().detach().float()

        # normalize data
        if init:
            self.images = self.images / 255

        if len(self.images.size()) == 3:
            self.images = torch.unsqueeze(self.images, dim=1)  # add channel dimension


    def __getitem__(self, index):

        # indexing
        x, y = self.images[index], self.labels[index]

        # transformations
        if self.transformations is not None:
            x = self.transformations(x)

        return x, y


    def __len__(self):
        return len(self.labels)


    """ Helper functions """
    def mean(self):
        return self.images.mean(dim=(0, 2, 3))


    def std(self):
        return self.images.std(dim=(0, 2, 3))


    """ Class functions """
    def _sample_helper(self, n_local_data, alpha, n_classes, **kwargs):

        # setup sample
        alpha = torch.full_like(torch.zeros(n_classes), fill_value=alpha)
        dist = Dirichlet(alpha)
        sample = dist.sample() * n_local_data

        # get random sample from sampled class proportions
        out_ind = torch.zeros(0)
        for class_i, class_num in enumerate(sample):

            # skip if none of class needed
            if class_num <= 0:
                continue

            class_ind = (self.labels == class_i).nonzero()

            shuffle_ind = torch.randperm(len(class_ind))
            shuffle_ind = shuffle_ind[:int(class_num)]

            sub_ind = class_ind[shuffle_ind]
            out_ind = torch.cat([out_ind, sub_ind])

        out_ind = out_ind.squeeze()

        return out_ind


    def sample(self, n_users, n_local_data, alpha, n_classes, **kwargs):
        return [self._sample_helper(n_local_data, alpha, n_classes) for i in range(n_users)]


    def poison_(
        self,
        model, target, n_batch,
        gpu_start, test=0, user_id=-1,
        **kwargs
        ):

        # remove target class if evaluation data
        if test:
            rm_indices = (self.labels == target)
            self.images = self.images[~rm_indices]
            self.labels = self.labels[~rm_indices]

        self.transformations = None
        self.labels = torch.full_like(self.labels, fill_value=target)

        loader = DataLoader(
            self,
            batch_size=n_batch,
            shuffle=False,
            num_workers=1,
            pin_memory=True
        )

        # use model to poison images
        for index, (images, labels) in enumerate(loader):
            images, labels = images.cuda(gpu_start), labels.cuda(gpu_start)

            start = n_batch*index
            end = min(start + n_batch, len(self.images))

            with torch.no_grad():
                self.images[start:end] = model(images, user_id)

        self.target = target
        return None


    def get_user_data(
        self,
        user_indices, m_user, user_id, model,
        p_pois, target, n_batch, gpu_start,
        **kwargs
        ):

        user_images, user_labels = self.images[user_indices.long()], self.labels[user_indices.long()]

        # implement poisoning for malicious users
        if m_user:

            # give random shuffle
            n_pois = int(p_pois * len(user_indices))
            shuffle = torch.randperm(len(user_labels))
            user_images, user_labels = user_images[shuffle], user_labels[shuffle]

            to_pois = Custom2dDataset(user_images[:n_pois], user_labels[:n_pois], transformations=None)
            to_pois.poison_(
                model, target, n_batch, gpu_start,
                user_id=user_id
            )
            #to_pois.view_imgs()

            user_images[:n_pois] = to_pois.images
            user_labels[:n_pois] = to_pois.labels

        return Custom2dDataset(user_images, user_labels, self.transformations)


""" Model """
class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(256, 120)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(120, 84)
        self.relu4 = nn.ReLU()
        self.fc3 = nn.Linear(84, 10)
        self.relu5 = nn.ReLU()

    def forward(self, x):
        y = self.conv1(x)
        y = self.relu1(y)
        y = self.pool1(y)
        y = self.conv2(y)
        y = self.relu2(y)
        y = self.pool2(y)
        y = y.view(y.size(0), -1)
        y = self.fc1(y)
        y = self.relu3(y)
        y = self.fc2(y)
        y = self.relu4(y)
        y = self.fc3(y)
        y = self.relu5(y)
        return y


# visible stamp
class BasicStamp(nn.Module):

    def __init__(
        self,
        n_malicious: int = 0, dba: bool = 0,  # number of users for stamp to be shared by or distributed between
        row_shift: int = 0, col_shift: int = 0,  # offset of image from upper left corner
        row_size: int = 4, col_size: int = 4,  # size of the stamp for EACH user
        row_gap: int = 0, col_gap: int = 0  # gap between placement of EACH user
        ):

        # setup
        super(BasicStamp, self).__init__()
        self.dba = dba
        self.row_shift, self.col_shift = row_shift, col_shift
        self.row_size, self.col_size = row_size, col_size

        # find grid to distribute stamp between
        if self.dba:
            assert n_malicious > 1, 'dba requested but multiple users are not being requested for n_malicious'
            root = math.isqrt(n_malicious)

            # grid distribution of stamp
            if (root ** 2 == n_malicious):
                self.user_rows, self.user_cols = root, root
            elif (root * (root + 1) == n_malicious):
                self.user_rows, self.user_cols = root, root + 1
            else:
                raise ValueError(
                    f'Input n_malicious={n_malicious} is not easily grid divisible. '
                    f'Some suggested values for n_malicious include {root ** 2}, {root * (root + 1)}, and {(root + 1) ** 2}. '
                )

            # adjust stamp size
            stamp_rows = user_rows * row_size
            stamp_cols = user_cols * col_size

        # if centralized attack with multiple attackers
        else:
            assert gap_x + gap_y == 0, 'gap between users specified, but attack is not distributed among users'

            # adjust stamp size
            user_rows, user_cols = 1, 1

        # get randomized stamp
        self.stamp = torch.ones_like(
            size=(user_rows * row_size, user_cols * col_size)
        ).uniform_()

        # objects for stamp distribution
        self.stamp_row_starts = row_size * np.arange(self.user_rows)
        self.stamp_col_starts = col_size * np.arange(self.user_cols)

        # save object to indicate stamp placements
        row_move, col_move = row_size + row_gap, col_size + col_gap
        self.input_row_starts = row_move * np.arange(self.user_rows) + self.row_shift
        self.input_col_starts = col_move * np.arange(self.user_cols) + self.col_shift


    def _forward_helper(self, x, user_id):

        # identify user's placement within the grid
        row, col = user_id // self.user_cols, user_id % self.user_cols

        input_row_start, input_col_start = self.input_row_starts[row], self.input_col_starts[col]
        input_row_end, input_col_end = input_row_start + self.row_size, input_col_start + self.col_size

        # define portion of stamp to be placed
        stamp_row_start, stamp_col_start = self.stamp_row_starts[row], self.stamp_col_starts[col]
        stamp_row_end, stamp_col_end = stamp_row_start + self.row_size, stamp_col_start + self.col_size

        assert (input_col_end <= x.size(-2) and input_row_end <= x.size(-1))
        to_stamp = self.stamp[input_col_start:input_col_end, input_row_start:input_row_end]
        x[..., input_col_start:input_col_end, input_row_start:input_row_end] = to_stamp

        return x


    def forward(self, x, user_id: int = -1):

        if not self.dba:
            col_stop, row_stop = self.col_shift + self.col_size, self.row_shift + self.row_size
            x[..., self.col_shift:col_stop, self.row_shift:row_stop] = sefl.stamp

        else:
            assert user_id >= 0, 'need to specify valid user_id for stamp distribution and location'
            x = self._forward_helper(x, user_id)

        return x


""" Training """
def evaluate_output(
    loader, model, cost, gpu=0,  # evaluate
    logger=None, title='',  # logging
    output=1
    ):

    # initializations
    model = model.eval()
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

    return (eval_loss, eval_acc, output_layers, output_labels)

