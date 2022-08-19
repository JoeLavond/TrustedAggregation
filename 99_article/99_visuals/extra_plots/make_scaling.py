# packages
import numpy as np
import matplotlib.pyplot as plt


# helper
def scaling(x, num_classes):

    B = 1 / num_classes

    index = (x <= 2 * B)
    output = np.zeros_like(x)
    output[index] = (B - np.abs(B - x[index])) / B

    return output


""" Main function HERE """
def main():

    x = np.linspace(start=0, stop=1, num=200)

    plt.figure()
    plt.plot(x, scaling(x, 3))
    plt.ylim(0, 1.1)
    plt.xlim(0, 1)
    plt.vlines((1 / 3), 0, 1, 'g', 'dashed')
    plt.text((1 / 3), 1.05, 'balanced', c='g')
    plt.title('D = 3')

    plt.savefig('scaling.png')
    plt.show()


if __name__ == "__main__":
    main()
