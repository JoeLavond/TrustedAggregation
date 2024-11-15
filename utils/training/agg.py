"""
Implementation of alternate aggregation methods for federated learning
Global median and global mean are used to aggregate model weights
Global trust is used to filter out malicious or unusual model weights

"""
# packages
from typing import *

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


# alternate backdoor defense aggregation methods
def global_median_(global_model: nn.Module, model_list: List[nn.Module], gpu: int = 0):
    """
    Update global model (in-place) with the elementwise median of suggested model weights

    Args:
        global_model (nn.Module): global model to be updated
        model_list (List[nn.Module]): list of models to be aggregated
        gpu (int): gpu to use for computation

    """

    # iterate over model weights simultaneously
    for (_, global_weights), *obj in zip(
            global_model.state_dict().items(),
            *[model.state_dict().items() for model in model_list]
    ):
        obj = [temp[1] for temp in obj]  # keep model weights not names

        stacked = torch.stack(obj).float()
        global_weights.copy_(
            torch.quantile(
                stacked, q=0.5, dim=0
            ).cuda(gpu).type(global_weights.dtype)  # return median weight across models
        )

    return None


def global_mean_(global_model: nn.Module, model_list: List[nn.Module], beta: float = 0.1, gpu: int = 0):
    """
    Update global model (in-place) with the trimmed mean of suggested model weights
    Trimmed mean is the elementwise mean with the top and bottom beta of data removed
    Used to filter out malicious or unusual model weights before aggregation

    Args:
        global_model (nn.Module): global model to be updated
        model_list (List[nn.Module]): list of models to be aggregated
        beta (float): fraction of data to be removed from top and bottom of data
        gpu (int): gpu to use for computation

    """

    assert 0 <= beta < 1 / 2, 'invalid value of beta outside of [0, 1/2)'

    # iterate over model weights simultaneously
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
        trimmed_mean /= ((1 - 2 * beta) * n)

        # replace global weights with elementwise trimmed mean
        global_weights.copy_(trimmed_mean.cuda(gpu))

    return None


# Fed Trust
def __center_model_(new: nn.Module, old: nn.Module):
    """
    Center new model weights by subtracting old model weights
    Used to compute the difference between two models
    Modify new model in-place

    Args:
        new (nn.Module): new model to be centered
        old (nn.Module): old model to be subtracted

    """
    with torch.no_grad():
        for (_, new_weight), (_, old_weight) in zip(
                new.state_dict().items(),
                old.state_dict().items()
        ):
            new_weight.copy_(new_weight - old_weight.detach().clone().to(new_weight.device))

    return None


def __get_trust_score(trusted_model: nn.Module, user_model: nn.Module) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute trust score and scaling factor for user model weights

    Args:
        trusted_model (nn.Module): model to be trusted
        user_model (nn.Module): model to be evaluated

    Return:
        Tuple[torch.Tensor, torch.Tensor]: trust score and scaling factor for user model weights

    """
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

    return F.relu(output), trusted_norm / user_norm


def global_trust_(global_model: nn.Module, trusted_model: nn.Module, model_list: List[nn.Module], eta: float = 1):
    """
    Update global model (in-place) with the trusted mean of suggested model weights
    Trust is computed as the cosine similarity between the trusted model and user model
    Weights are scaled by the trust score and the scaling factor
    Used to filter out malicious or unusual model weights before aggregation

    Args:
        global_model (nn.Module): global model to be updated
        trusted_model (nn.Module): model to be trusted
        model_list (List[nn.Module]): list of models to be aggregated
        eta (float): learning rate for aggregation

    """
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

    # iterate over model weights simultaneously
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
