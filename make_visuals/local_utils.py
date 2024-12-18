# packages
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing

""" Key Operations """
def min_mean_smooth(
    values,  # ----- values to smooth
    scale=2  # ----- scaling to perform to smoothed values
):

    """
    Function: Return average value since global minimum
    Usage: Smoothing technique proposed for FL backdoor defense
    """

    # iterate sequence of values and perform smoothing on history
    output = []
    for i in range(len(values)):
        sub_values = values[:(i + 1)]  # subset to history
        output.append(
            np.mean(sub_values[np.argmin(sub_values):])  # avg since min
        )

    # return scaled sequence
    return scale * np.array(output)


def quadratic_scaling(
    proportion,  # ----- proportion of class to compute scaling coef
    num_classes  # ----- size of classification problem
):

    """
    Function: Return simple scaling coef given class proportion
        No observations -> 0
        Balanced proportion -> 1
        2 * Balanced proportion -> 0
        Any value in between is scaled linearly based on location
    Usage: Create visual to show scaling given different proportions
    """

    B = 1 / num_classes  # balanced proportion of labels

    # scale all values further than double balanced to 0
    index = (proportion <= 2 * B)
    output = np.zeros_like(proportion)

    # scale values based on how far they are from balanced
    output[index] = -1 / (B ** 2) * proportion[index] * (proportion[index] - 2 * B)

    return output


""" Helper function """
def threshold_diagnostics(
    rounds,  # -------- vector of the communication rounds for the values
    values,  # -------- vector of values to compute defense diagnostics
    thresholds, # ----- trusted user cutoffs for user submissions
    ):

    # create vector to indicate whether model is accepted
    accepted = np.array(
        [v < thresholds[int(r - 1)] for r, v in zip(rounds, values)]  # need to subtract one as using cutoff from previous round
    )

    # get increasing array of unique rounds
    unique_rounds = np.unique(rounds)
    unique_rounds.sort()

    # iterate unique rounds and store quantiles
    round_output, running_output = [], []
    for index, r in enumerate(unique_rounds):

        # compute proportion of values below threshold
        round_output.append(
            np.mean(accepted[rounds == r])
        )
        running_output.append(
            np.mean(accepted[rounds <= r])
        )

    return np.array(round_output), np.array(running_output)


def get_quantiles(
    rounds,  # ------- vector of the communication rounds for the values
    values,  # ------- vector of values to get quantiles for
    running=0  # ----- should the quantiles be per round or historical
):

    """
    Function: Return the quantiles of values occuring in rounds
        1. Can get quantiles for each unique round
        2. Can return quantiles for all previous (including current) round
    Usage: Summarize federated learning user model statistics.
        Can have multiple benign/malicious users per round
        Intended to help understand threshold performance against users
    """

    # get increasing array of unique rounds
    unique_rounds = np.unique(rounds)
    unique_rounds.sort()

    # iterate unique rounds and store quantiles
    q1, q3 = [], []
    for r in unique_rounds:

        # determine rounds to calculate quantiles over and subset values
        sub_rounds = (rounds <= r) if running else (rounds == r)
        sub_values = values[sub_rounds]

        # compute quantiles for subsetted values
        q1.append(np.quantile(sub_values, 0.25))
        q3.append(np.quantile(sub_values, 0.75))

    return np.array(q1), np.array(q3)


def get_smoothing(
    values,  # ----- values to smooth
    window  # ------ window to compute moving average over
):

    """
    Function: Compute multiple smoothing techniques for values
        Moving average of given window size
        Exponential smoothing with multiplicative trend
    Usage: For comparison to global min-mean smoothing in trusted defense
    """

    # save smoothed output
    moving_avg, exp_smooth = [], []

    # iterate sequence of values and smooth based on history
    for i in range(len(values)):
        sub_values = values[:(i + 1)]  # subset to history

        # compute moving average of size window
        moving_avg.append(
            np.mean(
                sub_values[max([0, i - window + 1]):]  # stop at beginning
            )
        )

        # if multiple values fit exponential smoothing
        if len(sub_values) < 2:
            exp_smooth.append(values[0])

        else:

            # fit smoothing and return prediction for most recent value
            model = ExponentialSmoothing(
                sub_values,
                trend='mul',
                initialization_method='estimated'
            ).fit()
            exp_smooth.append(model.fittedvalues[-1])

    return np.array(moving_avg), np.array(exp_smooth)






