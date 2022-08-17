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
    fig, axarr = plt.subplots(ncols=4, sharey=True, figsize=(16, 4))

    plt.sca(axarr[0])
    plt.plot(x, scaling(x, 2))
    plt.ylim(0, 1.1)
    plt.xlim(0, 1)
    plt.vlines((1 / 2), 0, 1, 'g', 'dashed')
    plt.text((1 / 2), 1.05, 'balanced', c='g')
    plt.title('D = 2')

    plt.sca(axarr[1])
    plt.plot(x, scaling(x, 3))
    plt.ylim(0, 1.1)
    plt.xlim(0, 1)
    plt.vlines((1 / 3), 0, 1, 'g', 'dashed')
    plt.text((1 / 3), 1.05, 'balanced', c='g')
    plt.title('D = 3')

    plt.sca(axarr[2])
    plt.plot(x, scaling(x, 5))
    plt.ylim(0, 1.1)
    plt.xlim(0, 1)
    plt.vlines((1 / 5), 0, 1, 'g', 'dashed')
    plt.text((1 / 5), 1.05, 'balanced', c='g')
    plt.title('D = 5')

    plt.sca(axarr[3])
    plt.plot(x, trunc_linear(x, 10))
    plt.ylim(0, 1.1)
    plt.xlim(0, 1)
    plt.vlines((1 / 10), 0, 1, 'g', 'dashed')
    plt.text((1 / 10), 1.05, 'balanced', c='g')
    plt.title('D = 10')

    plt.savefig('scaling.png')
    plt.show()


if __name__ == "__main__":
    main()
