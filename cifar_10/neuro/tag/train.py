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
sys.path.insert(1, f'{Path.home()}/tag/')
from utils import data, setup
from utils.training import agg, atk, dist, eval, neuro, train
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
    parser.add_argument('--n_user_data', default=500, type=int)
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
    parser.add_argument('--n_epochs_pois', default=15, type=int)
    parser.add_argument('--lr_pois', default=0.01, type=float)
    # benign users
    parser.add_argument('--n_epochs', default=10, type=int)
    parser.add_argument('--lr', default=0.01, type=float)

    """ Data poisoning """
    # attack
    parser.add_argument('--neuro_p', default=0.1, type=float)
    parser.add_argument('--dba', default=0, type=int)
    parser.add_argument('--p_pois', default=0.1, type=float)
    parser.add_argument('--target', default=0, type=int)
    parser.add_argument('--row_size', default=4, type=int)
    parser.add_argument('--col_size', default=4, type=int)
    # defense
    parser.add_argument('--d_start', default=1, type=int)
    parser.add_argument('--d_scale', default=2, type=float)
    parser.add_argument('--d_smooth', default=1, type=int)
    parser.add_argument('--n_val_data', default=500, type=int)
    parser.add_argument('--alpha_val', default=10000, type=int)
    parser.add_argument('--remove_val', default=1, type=int)

    return parser.parse_args()


""" Main function HERE """
def main():

    """ Setup """
    args = get_args()

    # output
    args.out_path = os.path.join(
        ('distributed' if args.dba else 'centralized'),
        f'alpha{args.alpha}--alpha_val{args.alpha_val}',
        f'n_rounds{args.n_rounds}--d_start{args.d_start}--m_start{args.m_start}--n_malicious{args.n_malicious}'
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
    users_data_indices = train_data.sample(args.n_users - 1, args.n_user_data, args.alpha, args.n_classes)
    [val_data_indices] = train_data.sample(1, args.n_val_data, args.alpha_val, args.n_classes)
    users_data_indices.append(val_data_indices)

    val_data = train_data.get_user_data(
        val_data_indices, m_user=0, user_id=-1, model=None, **vars(args)
    )

    val_data_scaling = val_data.quadratic_scaling(args.n_classes)

    # store output
    output_val_ks = []
    output_val_ks.append(
        [-1] + val_data_scaling.tolist()
    )

    val_data.transformations = None
    clean_val_loader = DataLoader(
        val_data,
        batch_size=args.n_batch,
        shuffle=False,
        num_workers=1,
        pin_memory=True
    )


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

    # initialize neurotoxin masking
    NT = neuro.Neurotoxin(
        model=copy.deepcopy(global_model),
        p=args.neuro_p
    )

    # global
    output_global_acc = []

    # defense
    output_user_ks = []
    output_val_ks_all = []


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

    # validation
    (global_clean_val_loss, global_clean_val_acc, global_output_layer, _) = eval.evaluate_output(
        clean_val_loader, global_model, cost, args.gpu_start,
        logger=None, title='validation clean',
        output=1
    )

    """ Validation User """
    # copy global model and subset to user local data
    user_model = copy.deepcopy(global_model).cuda(args.gpu_start + 1)
    user_data = train_data.get_user_data(
        users_data_indices[-1], m_users[-1], -1, stamp_model, **vars(args)
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
        lr=args.lr, weight_decay=args.wd, momentum=args.wd
    )

    # train local model
    (user_train_loss, user_train_acc) = train.training(
        user_loader, user_model, cost, user_opt,
        args.n_epochs, args.gpu_start + 1,
        logger=(logger if args.print_all else None), print_all=args.print_all
    )

    # validation
    (user_clean_val_loss, user_clean_val_acc, user_output_layer, _) = eval.evaluate_output(
        clean_val_loader, user_model, cost, args.gpu_start + 1,
        logger=None, title='validation clean',
        output=1
    )

    val_ks = [
        round(
            dist.ks_div(global_output_layer[:, c], user_output_layer[:, c]), 3
        ) for c in range(global_output_layer.shape[-1])
    ]
    output_val_ks.append(
        [0] + val_ks
    )

    val_ks = np.array(val_ks) * val_data_scaling
    val_ks_max = max(val_ks)
    output_val_ks_all.append(val_ks_max)

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
    # fed learn main loop
    for r in range(args.n_rounds):

        # setup
        r += 1
        round_start = time.time()
        global_update = copy.deepcopy(global_model).cpu()
        with torch.no_grad():
            for name, weight in global_update.state_dict().items():
                weight.zero_()

        # select subset of users
        user_update_count = 0
        malicious_update_count = 0

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
            if m_user:
                (user_train_loss, user_train_acc) = neuro.nt_training(
                    user_loader, user_model, cost, user_opt,
                    args.n_epochs, args.gpu_start + 1,
                    logger=(logger if m_user or args.print_all else None), print_all=args.print_all,
                    nt_obj=NT
                )
            else:
                (user_train_loss, user_train_acc) = train.training(
                    user_loader, user_model, cost, user_opt,
                    args.n_epochs, args.gpu_start + 1,
                    logger=(logger if m_user or args.print_all else None), print_all=args.print_all
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


            """ External validation """
            # validation
            (user_clean_val_loss, user_clean_val_acc, user_output_layer, _) = eval.evaluate_output(
                clean_val_loader, user_model, cost, args.gpu_start + 1,
                logger=None, title='validation clean',
                output=1
            )

            # execute ks cutoff if defending
            user_ks = [
                round(
                    dist.ks_div(global_output_layer[:, c], user_output_layer[:, c]), 3
                ) for c in range(global_output_layer.shape[-1])
            ]
            output_user_ks.append(
                [m_user, r] + user_ks
            )

            user_ks_max = max(user_ks)

            if args.d_smooth:
                thresh = np.mean(output_val_ks_all[np.argmin(output_val_ks_all):])
            else:
                thresh = output_val_ks_all[-1]

            thresh = args.d_scale * thresh
            thresh = np.minimum(thresh, 1)

            user_update = (user_ks_max < thresh)
            if m_user:
                logger.info(
                    'User KS Max: %.4f, Thresh: %.4f, Update: %r',
                    user_ks_max, thresh, user_update
                )

            # send updates to global
            if ((r < args.d_start) or user_update):
                user_update_count += 1
                with torch.no_grad():
                    for (_, global_weights), (_, update_weights), (_, user_weights) in zip(
                        global_model.state_dict().items(),
                        global_update.state_dict().items(),
                        user_model.state_dict().items()
                    ):
                        update_weights += (user_weights.cpu() - global_weights.cpu())


        """ Global model training """
        # update global weights
        if (user_update_count > 0):
            with torch.no_grad():
                for (_, global_weights), (_, update_weights) in zip(
                    global_model.state_dict().items(),
                    global_update.state_dict().items()
                ):
                    try:
                        update_weights /= user_update_count
                    except:
                        update_weights.copy_((update_weights / user_update_count).long())

                    global_weights += update_weights.cuda(args.gpu_start)

                # update neurotoxin mask
                NT.update_mask_(copy.deepcopy(global_model))


        round_end = time.time()
        logger.info(
            '\n\n--- GLOBAL EVALUATIONS - Round: %d of %d, Time: %.1f ---',
            r, args.n_rounds, round_end - round_start
        )

        # validation
        (global_clean_val_loss, global_clean_val_acc, global_output_layer, _) = eval.evaluate_output(
            clean_val_loader, global_model, cost, args.gpu_start,
            logger=None, title='validation clean',
            output=1
        )

        """ Validation User """
        # copy global model and subset to user local data
        user_model = copy.deepcopy(global_model).cuda(args.gpu_start + 1)
        user_data = train_data.get_user_data(
            users_data_indices[-1], m_users[-1], -1, stamp_model, **vars(args)
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
            lr=args.lr, weight_decay=args.wd, momentum=args.wd
        )

        # train local model
        (user_train_loss, user_train_acc) = train.training(
            user_loader, user_model, cost, user_opt,
            args.n_epochs, args.gpu_start + 1,
            logger=(logger if args.print_all else None), print_all=args.print_all
        )

        # validation
        (user_clean_val_loss, user_clean_val_acc, user_output_layer, _) = eval.evaluate_output(
            clean_val_loader, user_model, cost, args.gpu_start + 1,
            logger=None, title='validation clean',
            output=1
        )

        val_ks = [
            round(
                dist.ks_div(global_output_layer[:, c], user_output_layer[:, c]), 3
            ) for c in range(global_output_layer.shape[-1])
        ]
        output_val_ks.append(
            [r] + val_ks
        )

        val_ks = np.array(val_ks) * val_data_scaling
        val_ks_max = max(val_ks)
        output_val_ks_all.append(val_ks_max)

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
    suffix = (
        f'--neuro_p{args.neuro_p}'
        + (f'--d_scale{args.d_scale}' if args.d_scale != 2. else '')
        + (f'--n_val_data{args.n_val_data}' if args.n_val_data != args.n_user_data else '')
        + ('--no_smooth' if not args.d_smooth else '')
        + ('--vgg' if not args.resnet else '')
    )

    output_global_acc = np.array(output_global_acc)
    np.save(
        os.path.join(args.out_path, 'data', f'output_global_acc{suffix}.npy'), output_global_acc
    )

    output_val_ks = np.array(output_val_ks)
    np.save(
        os.path.join(args.out_path, 'data', f'output_val_ks{suffix}.npy'), output_val_ks
    )

    output_user_ks = np.array(output_user_ks)
    np.save(
        os.path.join(args.out_path, 'data', f'output_user_ks{suffix}.npy'), output_user_ks
    )


if __name__ == "__main__":
    main()
