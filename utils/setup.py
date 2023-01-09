# packages
import logging
import numpy as np
import os
import random
import torch


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

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    return None
