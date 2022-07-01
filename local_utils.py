# packages
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

    def shannon_entropy(self, agg=1):

        # setup
        label_counts = torch.bincount(self.labels).cpu().numpy()
        label_props = label_counts / sum(label_counts)
        temp = label_props * np.log(label_props)

        if agg:
            return -1 * sum(temp)
        else:
            return -1 * len(temp) * temp

    def simpson_entropy(self):

        # setup
        label_counts = torch.bincount(self.labels).cpu().numpy()
        label_props = label_counts / sum(label_counts)

        return -1 * np.log(sum([x ** 2 for x in label_props]))

    def min_entropy(self):

        # setup
        label_counts = torch.bincount(self.labels).cpu().numpy()
        label_props = label_counts / sum(label_counts)

        entropy = -1 * np.log(max(label_props))

        return entropy


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


