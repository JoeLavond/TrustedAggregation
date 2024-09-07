# packages
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.distributions import Dirichlet
from torch.utils.data import Dataset, DataLoader


class Custom3dDataset(Dataset):
    """ Initial Setup """

    def __init__(self, images, labels, transformations=None, permute=1):

        # load data
        self.target = None
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

    def sample(self, n_users, n_local_data, alpha, n_classes, **kwargs):
        return [self._sample_helper(n_local_data, alpha, n_classes) for _ in range(n_users)]

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

            start = n_batch * index
            end = min(start + n_batch, len(self.images))

            with torch.no_grad():
                self.images[start:end] = model(images, user_id)

        self.target = target
        # self.view_imgs()
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


class Custom2dDataset(Custom3dDataset):
    """ Initial Setup """

    def __init__(self, images, labels, transformations=None, permute=1):

        # load data
        self.target = None
        if isinstance(images, torch.Tensor):
            self.labels = labels.clone().detach()
            self.images = images.clone().detach().float()

        else:

            # restructure (channel, height, width)
            self.labels = torch.tensor(labels)
            self.images = torch.tensor(images).float()

            # normalize data
            self.images = self.images / 255


        # add missing channel dimension
        if len(self.images.shape) == 3:
            self.images = self.images.unsqueeze(dim=1)
            self.images = self.images.repeat(1, 3, 1, 1)

        # ensure that channels are the first dimension
        elif permute:
            self.images = self.images.permute(dims=(0, 3, 1, 2))

        # transforms
        self.transformations = transformations
