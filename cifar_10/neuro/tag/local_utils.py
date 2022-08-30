# packages
import copy
import math
import numpy as np
import re
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
class CustomDataset(Dataset):

    """ Initial Setup """
    def __init__(self, images, labels, transformations=None):

        # load data
        if isinstance(images, torch.Tensor):
            self.images = images.clone().detach().float()
            self.labels = labels.clone().detach()

        else:

            # restructure (channel, height, width)
            self.images = torch.tensor(images).float().permute(dims=(0, 3, 1, 2))
            self.labels = torch.tensor(labels)

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
    # Channel is last dimension
    def mean(self):
        return self.images.mean(dim=(0, 2, 3))

    def std(self):
        return self.images.std(dim=(0, 2, 3))

    def view_imgs(self, n=2):

        for i in range(n):
            temp = self.images[i].squeeze().permute(1, 2, 0)
            plt.imshow(temp)
            plt.show()

        return None

    def get_class(self, class_i):

        class_index = (self.labels == class_i)

        return CustomDataset(
            self.images[class_index],
            self.labels[class_index],
            self.transformations
        )


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

            to_pois = CustomDataset(user_images[:n_pois], user_labels[:n_pois], transformations=None)
            to_pois.poison_(
                model, target, n_batch, gpu_start,
                user_id=user_id
            )
            #to_pois.view_imgs()

            user_images[:n_pois] = to_pois.images
            user_labels[:n_pois] = to_pois.labels

        return CustomDataset(user_images, user_labels, self.transformations)


    def view_classes(
        self,
        users_indices, m_users, model,
        p_pois, target, n_batch, gpu_start,
        n_classes, n_local_data,
        n=10, backwards=1, **kwargs
        ):

        for i in range(n):

            if backwards:
                users_indices = users_indices[-n:]
                m_users = m_users[-n:]
            else:
                users_indices = users_indices[:n]
                m_users = m_users[:n]

            users_data = [
                self.get_user_data(
                    u_indices, m_user, model,
                    p_pois, target, n_batch, gpu_start
                ) for (u_indices, m_user) in zip(users_indices, m_users)
            ]

            users_data = torch.stack(
                [torch.bincount(data.labels) for data in users_data]
            ).cpu().numpy()

            plt.figure()
            sns.heatmap(
                users_data,
                vmin=0, vmax=n_local_data, linewidths=.5,
                annot=True, fmt='d'
            )
            plt.show()

            return None


""" Model """
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
            for (_, new_weight), (_, old_weight) in zip(
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
        n_malicious, dba,
        size_x, size_y,
        gap_x, gap_y,
        shift_x=0, shift_y=0
        ):

        super(BasicStamp, self).__init__()

        # setup
        if dba:
            root = math.isqrt(n_malicious)
            assert n_malicious == root ** 2
        else:
            root, n_malicious = 2, 4

        # save stamp params
        self.dba = dba
        self.n_malicious, self.root = n_malicious, root
        self.size_x, self.size_y = size_x, size_y

        move_x, move_y = size_x + gap_x, size_y + gap_y
        self.x_start, self.y_start = move_x * np.arange(root) + shift_x, move_y * np.arange(root) + shift_y


    def _forward_helper(self, x, user_id):

        # setup
        row, col = user_id // self.root, user_id % self.root
        x_start, y_start = self.x_start[col], self.y_start[row]
        x_end, y_end = x_start + self.size_x, y_start + self.size_y

        assert (x_end <= x.size(-2) and y_end <= x.size(-1))
        x[..., y_start:y_end, x_start:x_end] = 1

        return x


    def forward(self, x, user_id=-1):

        if (not self.dba or user_id == -1):

            # apply global stamp
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


def _evaluate_conditional_helper(model, cost, gpu, data):

    # initializations
    n_eval = len(data)
    loader = DataLoader(
        data,
        batch_size=n_batch,
        shuffle=True,
        num_workers=1,
        pin_memory=True
    )

    # evaluate
    (eval_loss, eval_acc) = gu.evaluate(
        loader, model, cost, gpu
    )

    return (n_eval, eval_loss, eval_acc)


def evaluate_conditional(
    data, model, cost,
    n_classes=10, gpu_start=0,  # evaluate
    logger=None, title=''  # logging
    ):

    """
    return the number of images, average loss, and accuracy for each class
    """

    # initializations
    model = model.eval()
    frozen_helper = partial(
        _evaluate_conditional_helper,
        model, cost, gpu
    )

    # evaluation
    eval_start = time.time()
    helper_output = [
        _evaluate_conditional_helper(
            data.get_class(i)
        ) for i in range(args.n_classes)
    ]
    eval_end = time.time()

    # conditional summary
    full_output = [
        [helper_output[i][j] for i in range(args.n_classes)]
        for j in range(3)
    ]

    # traditional summary
    (n_total, total_loss, total_acc) = [
        sum(full_output[0]),
        sum([a*b for a,b in zip(full_output[0], full_output[1])]),
        sum([a*b for a,b in zip(full_output[0], full_output[2])])
    ]

    total_loss /= n_total
    total_acc /= n_total

    if logger is not None:
        logger.info(
            title.upper() + ' - Time %.1f, Loss %.4f, Acc %.4f',
            eval_end - eval_start, eval_loss, eval_acc
        )

    return full_output


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


