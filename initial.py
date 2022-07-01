""" Packages """
# base
import argparse
import copy
import logging
import os
import sys
import time

# numeric
import numpy as np
import scipy.stats as stats
from sklearn.model_selection import train_test_split

# visual
import matplotlib
#matplotlib.use('agg')
import matplotlib.pyplot as plt
#matplotlib.pyplot.switch_backend('agg')
import seaborn as sns

# torch
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms as T
from torch.utils.data import Dataset, DataLoader

# source
sys.path.insert(2, '/home/joe/')
import global_utils as gu
import local_utils as lu

sys.path.insert(2, '/home/joe/models/')
import resnet


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
    parser.add_argument('--print_all', default=0, type=int)

    # modeling
    parser.add_argument('--model_name', default='resnet', type=str)

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
    parser.add_argument('--n_epochs_pois', default=15, type=int)
    parser.add_argument('--lr_pois', default=0.01, type=float)
    # benign users
    parser.add_argument('--n_epochs', default=10, type=int)
    parser.add_argument('--lr', default=0.01, type=float)

    """ Data poisoning """
    # attack
    parser.add_argument('--dba', default=0, type=float)
    parser.add_argument('--p_pois', default=0.1, type=float)
    parser.add_argument('--target', default=0, type=int)
    parser.add_argument('--size_x', default=4, type=int)
    parser.add_argument('--size_y', default=1, type=int)
    parser.add_argument('--gap', default=1, type=int)
    # defense
    parser.add_argument('--d_start', default=1, type=int)
    parser.add_argument('--alpha_val', default=10000, type=int)
    parser.add_argument('--n_warmup', default=10, type=int)
    parser.add_argument('--p_warmup', default=.95, type=int)
    parser.add_argument('--remove_val', default=1, type=int)

    return parser.parse_args()


""" Main function HERE """
def main():

    """ Setup """
    args = get_args()

    # output
    args.out_path = (
        ('distributed' if args.dba else 'centralized')
        + '/alpha' + str(args.alpha) + '--alpha_val' + str(args.alpha_val)
        + '/n_rounds' + str(args.n_rounds) + '--d_start' + str(args.d_start) + '--m_start' + str(args.m_start)
    )
    if not os.path.exists(args.out_path):
        os.makedirs(args.out_path)

    gu.set_seeds(args.seed)
    logger = gu.get_log(args.out_path)
    logger.info(args)

    """ Training data """
    cifar_trans = T.Compose([
        T.Pad(padding=4),
        T.RandomHorizontalFlip(),
        T.RandomCrop(size=32)
    ])

    train_data = datasets.CIFAR10(
        root='/home/joe/data/',
        train=True,
        download=True
    )

    train_data = lu.CustomDataset(train_data.data, train_data.targets, cifar_trans)
    cifar_mean = train_data.mean()
    cifar_std = train_data.std()

    # get user data
    users_data_indices = train_data.sample(args.n_users - 1, args.n_local_data, args.alpha, args.n_classes)
    [val_data_indices] = train_data.sample(1, args.n_local_data, args.alpha_val, args.n_classes)
    users_data_indices.append(val_data_indices)

    val_data = train_data.get_user_data(
        val_data_indices, m_user=0, user_id=-1, model=None, **vars(args)
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
    stamp_model = lu.BasicStamp(
        args.n_malicious, args.dba,
        args.size_x, args.size_y,
        args.gap, args.gap
    ).cuda(args.gpu_start)
    stamp_model = stamp_model.eval()


    """ Federated learning setup """
    # initialize global model
    cost = nn.CrossEntropyLoss()
    global_model = nn.Sequential(
        gu.StdChannels(cifar_mean, cifar_std),
        resnet.resnet18(pretrained=False)
    ).cuda(args.gpu_start)
    global_model = global_model.eval()

    # global
    output_global_loss_clean, output_global_acc_clean = [], []
    output_global_loss_pois, output_global_acc_pois = [], []

    # defense
    output_val_ks_all, output_val_ks_cut = [], []
    output_benign_ks_all, output_benign_ks_q1, output_benign_ks_q3 = [], [], []
    output_malicious_ks_all, output_malicious_ks_mean, output_malicious_ks_min = [], [], []

    # other
    output_update_all, output_update_benign, output_update_malicious = 0, 0, 0
    output_update_prop, output_update_benign_prop = [], []

    output_sensitivity_all, output_specificity_all = [], []
    output_sensitivity_cum, output_specificity_cum = [], []

    """ Import testing data """
    # testing data
    test_data = datasets.CIFAR10(
        root='/home/joe/data/',
        train=False,
        download=True

    )

    clean_test_x, pois_test_x, clean_test_y, pois_test_y = train_test_split(
        np.array(test_data.data), np.array(test_data.targets),
        test_size=0.5, stratify=np.array(test_data.targets)

    )

    clean_test_data = lu.CustomDataset(clean_test_x, clean_test_y)
    clean_test_loader = DataLoader(
        clean_test_data,
        batch_size=args.n_batch,
        shuffle=False,
        num_workers=1,
        pin_memory=True

    )

    # poison subset of test data
    pois_test_data = lu.CustomDataset(pois_test_x, pois_test_y)
    pois_test_data.poison_(stamp_model, args.target, args.n_batch, args.gpu_start, test=1)
    pois_test_data.view_imgs()

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
    (global_clean_val_loss, global_clean_val_acc, global_output_layer, _) = lu.evaluate_output(
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
    (user_train_loss, user_train_acc) = gu.training(
        user_loader, user_model, cost, user_opt,
        args.n_epochs, args.gpu_start + 1,
        logger=(logger if args.print_all else None), print_all=args.print_all
    )

    # validation
    (user_clean_val_loss, user_clean_val_acc, user_output_layer, _) = lu.evaluate_output(
        clean_val_loader, user_model, cost, args.gpu_start + 1,
        logger=None, title='validation clean',
        output=1
    )

    val_ks = [
        round(
            lu.ks_div(global_output_layer[:, c], user_output_layer[:, c]), 3
        ) for c in range(global_output_layer.shape[-1])
    ]
    val_ks_max = max(val_ks)
    output_val_ks_all.append(val_ks_max)

    c_scale = args.n_warmup * (1 - args.p_warmup) / args.p_warmup
    d_scale = (1 / (1 + c_scale))

    output_val_ks_cut.append(
        2 * d_scale * val_ks_max
    )

    if args.print_all:
        logger.info([output_val_ks_cut[-1]] + val_ks)

    # testing
    (global_clean_test_loss, global_clean_test_acc) = gu.evaluate(
        clean_test_loader, global_model, cost, args.gpu_start,
        logger=logger, title='testing clean'
    )
    output_global_loss_clean.append(global_clean_test_loss)
    output_global_acc_clean.append(global_clean_test_acc)

    (global_pois_test_loss, global_pois_test_acc) = gu.evaluate(
        pois_test_loader, global_model, cost, args.gpu_start,
        logger=logger, title='testing pois'
    )
    output_global_loss_pois.append(global_pois_test_loss)
    output_global_acc_pois.append(global_pois_test_acc)


    """ Federated learning communication rounds """
    for r in range(args.n_rounds):
        r += 1

        # setup
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
            output_update_all += 1
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
            (user_train_loss, user_train_acc) = gu.training(
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


            """ External validation """
            # validation
            (user_clean_val_loss, user_clean_val_acc, user_output_layer, _) = lu.evaluate_output(
                clean_val_loader, user_model, cost, args.gpu_start + 1,
                logger=None, title='validation clean',
                output=1
            )

            # execute ks cutoff if defending
            user_ks = [
                round(
                    lu.ks_div(global_output_layer[:, c], user_output_layer[:, c]), 3
                ) for c in range(global_output_layer.shape[-1])
            ]
            user_ks_max = max(user_ks)

            user_update = (user_ks_max < output_val_ks_cut[-1])

            # store output
            if (m_user or args.print_all):
                logger.info([user_update] + user_ks)

            if m_user:

                output_update_malicious += user_update
                output_malicious_ks_all.append(user_ks_max)
                output_sensitivity_all.append(1 if not user_update else 0)

            else:

                output_update_benign += user_update
                output_benign_ks_all.append(user_ks_max)
                output_specificity_all.append(1 if user_update else 0)

            # evaluate
            if args.print_all:
                (global_clean_test_loss, global_clean_test_acc) = gu.evaluate(
                    clean_test_loader, global_model, cost, args.gpu_start,
                    logger=logger, title='testing clean'
                )

                (global_pois_test_loss, global_pois_test_acc) = gu.evaluate(
                    pois_test_loader, global_model, cost, args.gpu_start,
                    logger=logger, title='testing pois'
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
        # save output
        output_benign_ks_q3.append(np.quantile(output_benign_ks_all, 0.75))
        output_benign_ks_q1.append(np.quantile(output_benign_ks_all, 0.25))

        output_sensitivity_cum.append(np.mean(output_sensitivity_all))
        output_specificity_cum.append(np.mean(output_specificity_all))

        output_update_prop.append(output_update_benign / output_update_all)
        output_update_benign_prop.append(output_update_benign / (output_update_benign + output_update_malicious))

        try:
            output_malicious_ks_mean.append(np.mean(output_malicious_ks_all))
            output_malicious_ks_min.append(min(output_malicious_ks_all))
        except:
            pass

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

        round_end = time.time()
        logger.info(
            '\n\n--- GLOBAL EVALUATIONS - Round: %d of %d, Time: %.1f ---',
            r, args.n_rounds, round_end - round_start
        )

        # validation
        (global_clean_val_loss, global_clean_val_acc, global_output_layer, _) = lu.evaluate_output(
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
        (user_train_loss, user_train_acc) = gu.training(
            user_loader, user_model, cost, user_opt,
            args.n_epochs, args.gpu_start + 1,
            logger=(logger if args.print_all else None), print_all=args.print_all
        )

        # validation
        (user_clean_val_loss, user_clean_val_acc, user_output_layer, _) = lu.evaluate_output(
            clean_val_loader, user_model, cost, args.gpu_start + 1,
            logger=None, title='validation clean',
            output=1
        )

        val_ks = [
            round(
                lu.ks_div(global_output_layer[:, c], user_output_layer[:, c]), 3
            ) for c in range(global_output_layer.shape[-1])
        ]

        val_ks_max = max(val_ks)
        output_val_ks_all.append(val_ks_max)

        d_scale = ((r + 1) / ((r + 1) + c_scale))
        output_val_ks_cut.append(
            2 * d_scale * np.mean(output_val_ks_all[np.argmin(output_val_ks_all):])
        )

        if args.print_all:
            logger.info([output_val_ks_cut[-1]] + val_ks)

        # testing
        (global_clean_test_loss, global_clean_test_acc) = gu.evaluate(
            clean_test_loader, global_model, cost, args.gpu_start,
            logger=logger, title='testing clean'
        )
        output_global_loss_clean.append(global_clean_test_loss)
        output_global_acc_clean.append(global_clean_test_acc)

        (global_pois_test_loss, global_pois_test_acc) = gu.evaluate(
            pois_test_loader, global_model, cost, args.gpu_start,
            logger=logger, title='testing pois'
        )
        output_global_loss_pois.append(global_pois_test_loss)
        output_global_acc_pois.append(global_pois_test_acc)


    """ Visualizations """
    logging.disable()

    # defense
    if args.d_start < args.n_rounds:

        if args.m_start < args.n_rounds:

            plt.figure()
            plt.plot(range(args.n_rounds + 1), [2*x for x in output_val_ks_all])
            plt.plot(range(args.n_rounds + 1), output_val_ks_cut)
            if args.dba:
                plt.plot(range(args.m_start, args.n_rounds + 1), output_malicious_ks_mean)
            else:
                plt.plot(range(args.m_start, args.n_rounds + 1), output_malicious_ks_all)
            plt.plot(range(args.m_start, args.n_rounds + 1), output_malicious_ks_min)
            plt.vlines(args.d_start, -.05, 1, 'b', 'dashed')
            plt.text(args.d_start, 1.0667, 'd-start')
            plt.vlines(args.m_start, -.05, 1, 'r', 'dashed')
            plt.text(args.m_start, 1.0333, 'a-start')
            plt.xlabel('Round')
            plt.ylim(-.05, 1.1)
            plt.title('KS Cutoff Over Communication Rounds')
            plt.legend(labels=['cut-old', 'cut-new', 'malicious-all', 'malicious-min'])
            plt.savefig(os.path.join(args.out_path, 'defense_malicious.png'))

            fig, ax1 = plt.subplots()
            ax1.plot(range(1, args.n_rounds + 1), output_sensitivity_cum, 'g-')
            plt.vlines(args.d_start, -.05, 1, 'b', 'dashed')
            plt.text(args.d_start, 1.0667, 'd-start')
            ax2 = ax1.twinx()
            ax2.plot(range(1, args.n_rounds + 1), output_specificity_cum, 'b-')
            ax2.set_ylabel('Cumulative Specificity', c='b')
            ax2.set_ylim(ax1.get_ylim())
            plt.vlines(args.m_start, -.05, 1, 'r', 'dashed')
            plt.text(args.m_start, 1.0333, 'a-start')
            ax1.set_xlabel('Round')
            ax1.set_ylabel('Cumulative Sensitivity', c='g')
            ax1.set_ylim(-.05, 1.1)
            plt.title('Defense Success Metrics Over Communication Rounds')
            plt.savefig(os.path.join(args.out_path, 'defense_eval.png'))

        plt.figure()
        plt.plot(range(args.n_rounds + 1), [2*x for x in output_val_ks_all])
        plt.plot(range(args.n_rounds + 1), output_val_ks_cut)
        plt.plot(range(1, args.n_rounds + 1), output_benign_ks_q1)
        plt.plot(range(1, args.n_rounds + 1), output_benign_ks_q3)
        plt.vlines(args.d_start, -.05, 1, 'b', 'dashed')
        plt.text(args.d_start, 1.0667, 'd-start')
        if args.m_start < args.n_rounds:
            plt.vlines(args.m_start, -.05, 1, 'r', 'dashed')
            plt.text(args.m_start, 1.0333, 'a-start')
        plt.xlabel('Round')
        plt.ylim(-.05, 1.1)
        plt.title('KS Cutoff Over Communication Rounds')
        plt.legend(labels=['cut-old', 'cut-new', 'benign-low', 'benign-high'])
        plt.savefig(os.path.join(args.out_path, 'defense_benign.png'))

    # global
    fig, ax1 = plt.subplots()
    ax1.plot(range(args.n_rounds + 1), output_global_acc_clean, 'g-')
    plt.vlines(args.d_start, -.05, 1, 'b', 'dashed')
    plt.text(args.d_start, 1.0667, 'd-start')
    if args.m_start < args.n_rounds:
        ax2 = ax1.twinx()
        ax2.plot(range(args.n_rounds + 1), output_global_acc_pois, 'r-')
        ax2.set_ylabel('Attack Success Rate', c='r')
        ax2.set_ylim(ax1.get_ylim())
        plt.vlines(args.m_start, -.05, 1, 'r', 'dashed')
        plt.text(args.m_start, 1.0333, 'a-start')
    ax1.set_xlabel('Round')
    ax1.set_ylabel('Correct Classification Rate', c='g')
    ax1.set_ylim(-.05, 1.1)
    plt.title('Testing Sets Evaluated Over Communication Rounds')
    plt.savefig(os.path.join(args.out_path, 'global_acc_eval.png'))

    fig, ax1 = plt.subplots()
    ax1.plot(range(1, args.n_rounds + 1), output_update_prop, 'g-')
    plt.vlines(args.d_start, -.05, 1, 'b', 'dashed')
    plt.text(args.d_start, 1.0667, 'd-start')
    if args.m_start < args.n_rounds:
        ax2 = ax1.twinx()
        ax2.plot(range(1, args.n_rounds + 1), output_update_benign_prop, 'b-')
        ax2.set_ylabel('Benign Update Rate', c='b')
        ax2.set_ylim(ax1.get_ylim())
        plt.vlines(args.m_start, -.05, 1, 'r', 'dashed')
        plt.text(args.m_start, 1.0333, 'a-start')
    ax1.set_xlabel('Round')
    ax1.set_ylabel('Update Rate', c='g')
    ax1.set_ylim(-.05, 1.1)
    plt.title('Global Update Statistics Over Communication Rounds')
    plt.savefig(os.path.join(args.out_path, 'global_update_eval.png'))



if __name__ == "__main__":
    main()