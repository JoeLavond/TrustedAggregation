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
from sklearn.model_selection import train_test_split

# torch
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms as T
from torch.utils.data import DataLoader

# source
sys.path.insert(1, f'{Path.home()}/tag')
from utils import data, setup
from utils.training import agg, atk, eval, train
from utils.modeling import pre, resnet, vgg


""" Setup """
# file control
def get_args():
    parser = argparse.ArgumentParser()

    """ File setup """
    # control
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--n_classes', default=10, type=int)
    parser.add_argument('--gpu_start', default=0, type=int)
    # output
    parser.add_argument('--resnet', default=1, type=int)
    parser.add_argument('--print_all', default=0, type=int)

    """ Federated learning """
    # basic fl
    parser.add_argument('--n_users', default=100, type=int)
    parser.add_argument('--n_local_data', default=500, type=int)
    parser.add_argument('--p_report', default=0.1, type=float)
    parser.add_argument('--n_rounds', default=1, type=int)
    parser.add_argument('--alpha', default=10000, type=int)
    # all users
    parser.add_argument('--n_batch', default=64, type=int)
    parser.add_argument('--mom', default=0.9, type=float)
    parser.add_argument('--wd', default=5e-4, type=float)
    # malicious users
    parser.add_argument('--m_start', default=1, type=int)
    parser.add_argument('--m_scale', default=1, type=int)
    parser.add_argument('--p_malicious', default=None, type=float)
    parser.add_argument('--n_malicious', default=1, type=int)
    parser.add_argument('--n_epochs_pois', default=20, type=int)
    parser.add_argument('--lr_pois', default=0.01, type=float)
    # benign users
    parser.add_argument('--n_epochs', default=10, type=int)
    parser.add_argument('--lr', default=0.01, type=float)

    """ Data poisoning """
    # attack
    parser.add_argument('--dba', default=0, type=float)
    parser.add_argument('--p_pois', default=0.1, type=float)
    parser.add_argument('--target', default=0, type=int)
    parser.add_argument('--row_size', default=4, type=int)
    parser.add_argument('--col_size', default=4, type=int)
    # defense
    parser.add_argument('--alpha_val', default=10000, type=int)
    parser.add_argument('--remove_val', default=1, type=int)
    parser.add_argument('--trim_mean', default=0, type=int)
    parser.add_argument('--beta', default=0.1, type=float)

    return parser.parse_args()


""" Main function HERE """
def main():

    """ Setup """
    args = get_args()

    # output
    args.out_path = (
        ('mean' if args.trim_mean else 'median')  # agg method
        + ('/distributed' if args.dba else '/centralized')
        + '/alpha' + str(args.alpha) + '--alpha_val' + str(args.alpha_val)
        + '/n_rounds' + str(args.n_rounds)
        + '--m_start' + str(args.m_start) + '--n_malicious' + str(args.n_malicious)
    )
    args.suffix = (
        (f'--beta{args.beta}' if args.trim_mean else '')
        + ('--vgg' if not args.resnet else '')
    )

    if not os.path.exists(args.out_path):
        os.makedirs(args.out_path)
    if not os.path.exists(os.path.join(args.out_path, 'data')):
        os.makedirs(
            os.path.join(args.out_path, 'data')
        )

    setup.set_seeds(args.seed)
    logger = setup.get_log(args.out_path)
    logger.info(args)

    """ Training data """
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

    train_data = data.Custom3dDataset(train_data.data, train_data.targets, cifar_trans)
    cifar_mean = train_data.mean()
    cifar_std = train_data.std()

    # get user data
    users_data_indices = train_data.sample(args.n_users - 1, args.n_local_data, args.alpha, args.n_classes)
    [val_data_indices] = train_data.sample(1, args.n_local_data, args.alpha_val, args.n_classes)
    users_data_indices.append(val_data_indices)

    """ Data poisoning """
    # establish malicious users
    if args.p_malicious:
        args.n_malicious = int(args.n_users * args.p_malicious)

    m_users = torch.zeros(args.n_users)
    m_users[0:args.n_malicious] = 1

    # define trigger model
    stamp_model = atk.BasicStamp(
        args.n_malicious, args.dba,
        row_size=args.row_size, col_size=args.col_size
    ).cuda(args.gpu_start)
    stamp_model = stamp_model.eval()


    """ Federated learning setup """
    # initialize global model
    cost = nn.CrossEntropyLoss()
    global_model = nn.Sequential(
        pre.StdChannels(cifar_mean, cifar_std),
        (
            resnet.resnet18(num_classes=args.n_classes, pretrained=False)
            if args.resnet else vgg.vgg16_bn()
        )
    ).cuda(args.gpu_start)
    global_model = global_model.eval()

    # global
    output_global_acc = []


    """ Import testing data """
    # testing data
    test_data = datasets.CIFAR10(
        root=f'{Path.home()}/data/',
        train=False,
        download=True
    )

    clean_test_x, pois_test_x, clean_test_y, pois_test_y = train_test_split(
        np.array(test_data.data), np.array(test_data.targets),
        test_size=0.5, stratify=np.array(test_data.targets)

    )

    clean_test_data = data.Custom3dDataset(clean_test_x, clean_test_y)
    clean_test_loader = DataLoader(
        clean_test_data,
        batch_size=args.n_batch,
        shuffle=False,
        num_workers=1,
        pin_memory=True

    )

    # poison subset of test data
    pois_test_data = data.Custom3dDataset(pois_test_x, pois_test_y)
    pois_test_data.poison_(stamp_model, args.target, args.n_batch, args.gpu_start, test=1)

    pois_test_loader = DataLoader(
                pois_test_data,
                batch_size=args.n_batch,
                shuffle=False,
                num_workers=1,
                pin_memory=True

    )

    """ Initial validation """
    logger.info(
        '\n\n--- GLOBAL VALIDATION - Round: %d of %d ---',
        0, args.n_rounds
    )

    # testing
    (global_clean_test_loss, global_clean_test_acc) = eval.evaluate(
        clean_test_loader, global_model, cost, args.gpu_start,
        logger=logger, title='testing clean'
    )

    (global_pois_test_loss, global_pois_test_acc) = eval.evaluate(
        pois_test_loader, global_model, cost, args.gpu_start,
        logger=logger, title='testing pois'
    )

    output_global_acc.append(
        [0, global_clean_test_acc, global_pois_test_acc]
    )


    """ Federated learning communication rounds """
    for r in range(args.n_rounds):
        r += 1

        # setup
        round_start = time.time()
        global_updates = []

        user_update_count = 0
        malicious_update_count = 0

        # select subset of users
        if (r < args.m_start):

            user_subset_index = torch.randperm(args.n_users - args.n_malicious - args.remove_val)[:int(args.n_users * args.p_report)]
            user_subset_index += args.n_malicious  # no malicious users

        else:

            # force malicious users in subset if past round start
            user_subset_index = torch.randperm(args.n_users - args.n_malicious - args.remove_val)[:int(args.n_users * args.p_report)]
            user_subset_index += args.n_malicious
            user_subset_index = [i if i < args.n_malicious else user_subset_index[i] for i in range(len(user_subset_index))]


        """ Local model training """
        for i, user_id in enumerate(user_subset_index):

            # setup
            user_indices = users_data_indices[user_id]
            m_user = m_users[user_id]

            if (m_user or args.print_all):
                logger.info(
                    '\n\n--- LOCAL TRAINING - Local Update: %d of %d, User: %d, Malicious: %s ---',
                    i + 1, len(user_subset_index), user_id, 'Yes' if m_user else 'No'
                )

            # copy global model and subset to user local data
            user_model = copy.deepcopy(global_model).cuda(args.gpu_start + 1)
            user_data = train_data.get_user_data(
                user_indices, m_user, user_id, stamp_model, **vars(args)
            )

            user_loader = DataLoader(
                user_data,
                batch_size=args.n_batch,
                shuffle=True,
                num_workers=1,
                pin_memory=True
            )

            user_opt = optim.SGD(
                user_model.parameters(),
                lr=args.lr_pois if m_user else args.lr,
                weight_decay=args.wd, momentum=args.wd
            )

            # train local model
            (user_train_loss, user_train_acc) = train.training(
                user_loader, user_model, cost, user_opt,
                args.n_epochs_pois if m_user else args.n_epochs, args.gpu_start + 1,
                logger=(logger if (m_user or args.print_all) else None), print_all=args.print_all
            )

            # malicious scaling of model weights
            if (m_user and (args.m_scale != 1)):
                with torch.no_grad():
                    for (_, global_weights), (_, user_weights) in zip(
                        global_model.state_dict().items(),
                        user_model.state_dict().items()
                    ):
                        global_weights = global_weights.cuda(args.gpu_start + 1)
                        user_weights.copy_(
                            args.m_scale * (user_weights - global_weights) + global_weights
                        )

            # send updates to global
            global_updates.append(user_model.cpu())


        """ Global model training """
        # update global weights
        if args.trim_mean:
            agg.global_mean_(global_model, global_updates, args.beta)
        else:
            agg.global_median_(global_model, global_updates)

        round_end = time.time()
        logger.info(
            '\n\n--- GLOBAL EVALUATIONS - Round: %d of %d, Time: %.1f ---',
            r, args.n_rounds, round_end - round_start
        )

        # testing
        (global_clean_test_loss, global_clean_test_acc) = eval.evaluate(
            clean_test_loader, global_model, cost, args.gpu_start,
            logger=logger, title='testing clean'
        )

        (global_pois_test_loss, global_pois_test_acc) = eval.evaluate(
            pois_test_loader, global_model, cost, args.gpu_start,
            logger=logger, title='testing pois'
        )

        output_global_acc.append(
            [0, global_clean_test_acc, global_pois_test_acc]
        )


    """ Save output """
    output_global_acc = np.array(output_global_acc)
    np.save(
        os.path.join(args.out_path, 'data', f'output_global_acc{args.suffix}.npy'), output_global_acc
    )


if __name__ == "__main__":
    main()
