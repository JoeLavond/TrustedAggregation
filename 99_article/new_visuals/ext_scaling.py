# packages
import numpy as np
import matplotlib.pyplot as plt


# helper
def scaling(
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
    output[index] = (B - np.abs(B - proportion[index])) / B

    return output


""" Main function HERE """
def main():

    num_classes = 3
    x = np.linspace(start=0, stop=1, num=200)  # class proportions

    # create visual for k-way classification problem
    plt.figure()
    plt.plot(x, scaling(x, num_classes))
    plt.ylim(0, 1.1)
    plt.xlim(0, 1)
    plt.vlines((1 / num_classes), 0, 1, 'g', 'dashed')
    plt.text((1 / num_classes), 1.05, 'balanced', c='g')
    plt.title(f'D = {num_classes}')

    plt.savefig('visuals/scaling.png')
    plt.show()


if __name__ == "__main__":
    main()
