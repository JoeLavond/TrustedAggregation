# packages
import copy
import math
import numpy as np
from pathlib import Path
import re
import sys
import time
from functools import partial

import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.distributions.dirichlet import Dirichlet

# source
sys.path.insert(2, f'{Path.home()}')
import global_utils as gu


""" Data """
class Custom3dDataset(Dataset):

    """ Initial Setup """
    def __init__(self, images, labels, transformations=None, permute=1):

        # load data
        if isinstance(images, torch.Tensor):
            self.labels = labels.clone().detach()
            self.images = images.clone().detach().float()

        else:

            # restructure (channel, height, width)
            self.labels = torch.tensor(labels)
            self.images = torch.tensor(images).float()
            if permute:
                self.images = self.images.permute(dims=(0, 3, 1, 2))

            # normalize data
            self.images = self.images / 255

        # transforms
        self.transformations = transformations


    def __getitem__(self, index):

        # indexing
        x, y = self.images[index], self.labels[index]

        # transformations
        if self.transformations is not None:
            x = self.transformations(x)

        return x, y


    def __len__(self):
        return len(self.images)


    """ Helper functions """
    def mean(self):
        return self.images.mean(dim=(0, 2, 3))


    def std(self):
        return self.images.std(dim=(0, 2, 3))


    def view_imgs(self, n=1):

        for i in range(n):
            temp = self.images[i].squeeze().permute(1, 2, 0)
            plt.imshow(temp)
            plt.show()

        return None


    def linear_scaling(self, n_classes):

        B = 1 / n_classes  # proportion of perfectly balanced class

        # setup
        label_counts = torch.bincount(self.labels).cpu().numpy()
        label_props = label_counts / sum(label_counts)

        index = (label_props <= 2 * B)
        output = np.zeros_like(label_props)
        output[index] = (B - np.abs(B - label_props[index])) / B

        return output


    def quadratic_scaling(self, n_classes):

        """
        Function: Return fitted parabola and output given class prop as input
            Parabola passes through (0, 0), (2B, 0), and (B, 1)
            y = a * x * (x - 2B)
            1 = a * B * (B - 2B)
            1 = a * B * (B - 2B)
            1 = a * -1 * B ** 2
            a = -1 / (B ** 2)
        """

        B = 1 / n_classes  # proportion of perfectly balanced class

        # setup
        label_counts = torch.bincount(self.labels).cpu().numpy()
        label_props = label_counts / sum(label_counts)

        index = (label_props <= 2 * B)
        output = np.zeros_like(label_props)
        output[index] = -1 / (B ** 2) * label_props[index] * (label_props[index] - 2 * B)

        return output


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
        #self.view_imgs()
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

            to_pois = Custom3dDataset(user_images[:n_pois], user_labels[:n_pois], transformations=None)
            to_pois.poison_(
                model, target, n_batch, gpu_start,
                user_id=user_id
            )

            user_images[:n_pois] = to_pois.images
            user_labels[:n_pois] = to_pois.labels

        return Custom3dDataset(user_images, user_labels, self.transformations)


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


    def view_imgs(self, n=1):

        for i in range(n):
            temp = self.images[i].squeeze()
            plt.imshow(temp)
            plt.show()

        return None


    def linear_scaling(self, n_classes):

        B = 1 / n_classes  # proportion of perfectly balanced class

        # setup
        label_counts = torch.bincount(self.labels).cpu().numpy()
        label_props = label_counts / sum(label_counts)

        index = (label_props <= 2 * B)
        output = np.zeros_like(label_props)
        output[index] = (B - np.abs(B - label_props[index])) / B

        return output


    def quadratic_scaling(self, n_classes):

        """
        Function: Return fitted parabola and output given class prop as input
            Parabola passes through (0, 0), (2B, 0), and (B, 1)
            y = a * x * (x - 2B)
            1 = a * B * (B - 2B)
            1 = a * B * (B - 2B)
            1 = a * -1 * B ** 2
            a = -1 / (B ** 2)
        """

        B = 1 / n_classes  # proportion of perfectly balanced class

        # setup
        label_counts = torch.bincount(self.labels).cpu().numpy()
        label_props = label_counts / sum(label_counts)

        index = (label_props <= 2 * B)
        output = np.zeros_like(label_props)
        output[index] = -1 / (B ** 2) * label_props[index] * (label_props[index] - 2 * B)

        return output



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
        #self.view_imgs()
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


# alternate backdoor defense aggregation methods
def global_median_(global_model, model_list, gpu=0):
    """
    Function: Update global model (in-place) with the elementwise median of suggested model weights
    Usage: Model filtering of users for federated learning
    """

    # iterate over model weights simulataneously
    for (_, global_weights), *obj in zip(
        global_model.state_dict().items(),
        *[model.state_dict().items() for model in model_list]
    ):
        obj = [temp[1] for temp in obj]  # keep model weights not names

        stacked = torch.stack(obj).float()
        global_weights.copy_(
            torch.quantile(stacked, q=0.5, dim=0).cuda(gpu).type(global_weights.dtype)  # return median weight across models
        )

    return None


def global_mean_(global_model, model_list, beta=.1, gpu=0):
    """
    Function: Update global model (in-place) with the trimmed mean of suggested model weights
        Trimmed mean is the elementwise mean with the top and bottom beta of data removed
    Usage: Model filtering of users for federated learning
    """

    assert 0 <= beta and beta < 1/2, 'invalid value of beta outside of [0, 1/2)'

    # iterate over model weights simulataneously
    for (_, global_weights), *obj in zip(
        global_model.state_dict().items(),
        *[model.state_dict().items() for model in model_list]
    ):

        # setup
        obj = [temp[1] for temp in obj]  # keep model weights not names

        n = len(obj)
        k = int(n * beta)  # how many models is beta of model_list

        # remove beta largest and smallest entries elementwise
        stacked = torch.stack(obj).float()
        obj_sorted, _ = torch.sort(stacked, dim=0)
        trimmed = obj_sorted[k:(n - k)]

        # compute trimmed mean
        trimmed_mean = trimmed.sum(dim=0)
        trimmed_mean /= ((1 - 2*beta) * n)

        # replace global weights with elementwise trimmed mean
        global_weights.copy_(trimmed_mean)

    return None


""" Fed Trust """
def __center_model_(new, old):

    with torch.no_grad():
        for (_, new_weight), (_, old_weight) in zip(
            new.state_dict().items(),
            old.state_dict().items()
        ):
            new_weight.copy_(new_weight - old_weight.detach().clone().to(new_weight.device))

    return None


def __get_trust_score(trusted_model, user_model):

    """ Returns (relu trust score times the ratio of trusted to user norm) """

    # flatten all model weights to
    trusted_weights = []
    user_weights = []

    # iterate over model weights
    for (_, trusted_weight), (_, user_weight) in zip(
        trusted_model.state_dict().items(),
        user_model.state_dict().items()
    ):

        trusted_weights.append(torch.flatten(trusted_weight))
        user_weights.append(torch.flatten(user_weight))

    # cast to tensor
    trusted_weights = torch.cat(trusted_weights)
    user_weights = torch.cat(user_weights)

    # compute cosine similarity
    trusted_norm = torch.norm(trusted_weights)
    user_norm = torch.norm(user_weights)
    output = F.cosine_similarity(
        trusted_weights, user_weights, dim=0
    )

    return (F.relu(output), trusted_norm / user_norm)


def global_trust_(global_model, trusted_model, model_list, eta=1):

    # get model differences
    __center_model_(trusted_model, global_model)
    for m in model_list:
        __center_model_(m, global_model)

    # use helper function to get trust scores and weightings
    trust_obj = [
        __get_trust_score(trusted_model, u_model)
        for u_model in model_list
    ]
    trust_scores = [t[0] for t in trust_obj]
    trust_scales = [t[1] for t in trust_obj]

    # unpack helper functions
    trust_scores = torch.from_numpy(np.array(trust_scores))
    trust_scales = torch.from_numpy(np.array(trust_scales))

    # iterate over model weights simulataneously
    for (_, global_weights), *obj in zip(
        global_model.state_dict().items(),
        *[model.state_dict().items() for model in model_list]
    ):

        # scale by trust score
        trusted_obj = [
            score * scale * temp[1]
            for score, scale, temp in zip(trust_scores, trust_scales, obj)
        ]
        trusted_obj = torch.stack(trusted_obj)

        trusted_mean = trusted_obj.sum(dim=0)
        trusted_mean /= trust_scores.sum().item()

        # replace global weights with trusted mean
        global_weights.copy_(global_weights + eta * trusted_mean.to(global_weights.device))

    return None


# implement Neurotoxin attack using model weights instead of gradients
class Neurotoxin:

    """
    Function:
        Compute model weight changes
        Find locations of p*100% of the largest values
        Create mask to identify largest locations
        Given update, replace largest locations with orignal values
    Usage:
        Modification of Neurotoxin attack based on weights
    Remarks:
        NEED TO COPY.DEEPCOPY(MODEL) AS INPUT TO MODEL
        EXCEPT WHEN APPLYING THE MASK WITH MASK_()
    """

    def __init__(self,
        model, p
     ):

        # initializations
        self.p = p

        self.old_model = model.cpu()
        with torch.no_grad():
            for (_, weight) in self.old_model.state_dict().items():
                weight.zero_()

        # create mask from random initialized model
        self.update_mask_(model)


    def update_mask_(self, new_model):

        """
        Function: Create a flattened mask of model weights
            Omit positions of largest changes in model weights
        Usage: Perform Neurotoxin attack
            Project attack onto weights not regularly updated
            Intended to produce lasting attacks
        """

        # initializations
        weight_diffs = []
        weight_count = 0

        with torch.no_grad():
            for (name, new_weight), (_, old_weight) in zip(
                new_model.state_dict().items(),
                self.old_model.state_dict().items()
            ):

                ## skip non-float type weights (ex. batch-norm)
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

            ## find k weights to mask from p
            k = int(self.p * weight_count)

            # concatenate weights to find top k
            weight_diffs = torch.cat(weight_diffs)
            (_, top_indices) = torch.topk(weight_diffs, k)

            # create mask from top k indices
            self.mask = torch.zeros_like(weight_diffs, dtype=torch.int64)
            self.mask[top_indices] = 1  # replace top changes

            # move new model to old model
            self.old_model = new_model.cpu()


    def mask_(self, model):

        """
        Function: Use mask on proposed model update
            Replace locations of largest change
            Use instead values from original model
        Usage: Apply after every model update when using Neurotoxin
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
        self.n_malicious, self.dba = n_malicious, dba
        self.row_shift, self.col_shift = row_shift, col_shift
        self.row_size, self.col_size = row_size, col_size

        # find grid to distribute stamp between
        if self.dba:
            assert self.n_malicious > 1, 'dba requested but multiple users are not being requested for n_malicious'
            root = math.isqrt(self.n_malicious)

            # grid distribution of stamp
            if (root ** 2 == self.n_malicious):
                self.user_rows, self.user_cols = root, root
            elif (root * (root + 1) == self.n_malicious):
                self.user_rows, self.user_cols = root, root + 1
            else:
                raise ValueError(
                    f'Input n_malicious={self.n_malicious} is not easily grid divisible. '
                    f'Some suggested values for n_malicious include {root ** 2}, {root * (root + 1)}, and {(root + 1) ** 2}. '
                )

            # adjust stamp size
            stamp_rows = self.user_rows * row_size
            stamp_cols = self.user_cols * col_size

        # if centralized attack with multiple attackers
        else:
            assert row_gap + col_gap == 0, 'gap between users specified, but attack is not distributed among users'

            # adjust stamp size
            self.user_rows, self.user_cols = 1, 1

        # generate stamp
        self.stamp = torch.zeros(
            (self.user_rows * row_size, self.user_cols * col_size)
        )

        # create stamp pattern
        i, j = self.stamp.shape
        i = i // 4
        j = j // 4
        self.stamp[i:(3 * i), :] = 1
        self.stamp[:, j:(3 * j)] = 1

        # self.stamp = self.stamp.uniform_()  # get randomized stamp

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

        assert (input_row_end <= x.size(-1) and input_col_end <= x.size(-2))
        to_stamp = self.stamp[input_row_start:input_row_end, input_col_start:input_col_end]
        x[..., input_row_start:input_row_end, input_col_start:input_col_end] = to_stamp

        return x


    def forward(self, x, user_id: int = -1):
        assert user_id >= -1, 'need to specify valid user_id for stamp distribution and location'

        if not self.dba:
            col_stop, row_stop = self.col_shift + self.col_size, self.row_shift + self.row_size
            x[..., self.col_shift:col_stop, self.row_shift:row_stop] = self.stamp

        else:

            if user_id == -1:
                for i in range(self.n_malicious):
                    x = self._forward_helper(x, i)

            else:
                x = self._forward_helper(x, user_id)

        return x


""" Training """
def nt_training(
    loader, model, cost, opt, n_epochs=1, gpu=0, scheduler=None,  # training
    logger=None, title='training', print_all=0,  # logging
    nt_obj=None  # add neurotoxin attack
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

        if ((logger is not None) and (print_all or (epoch == n_epochs))):
            logger.info(
                title.upper() + ' - Epoch: %d, Time %.1f, Loss %.4f, Acc %.4f',
                epoch, train_end - train_start, train_loss, train_acc
            )

    return (train_loss, train_acc)


# evaluation that saves output distribution layer and labels
def evaluate_output(
    loader, model, cost, gpu=0,  # evaluate
    logger=None, title='',  # logging
    output=1
    ):

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

    return (eval_loss, eval_acc, output_layers, output_labels)


""" Divergence """
def empirical_cdfs(x, y):

    values = np.append(
        np.unique(x),
        np.unique(y)
    )
    values = np.sort(np.unique(values))

    x_cdf = np.array([np.sum(x <= val) / len(x) for val in values])
    y_cdf = np.array([np.sum(y <= val) / len(y) for val in values])

    return (x_cdf, y_cdf, values)


def ks_div(x, y):
    (x_cdf, y_cdf, _) = empirical_cdfs(x, y)
    return np.max(np.abs(y_cdf - x_cdf))


