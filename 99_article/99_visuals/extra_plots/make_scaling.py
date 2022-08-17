# packages
import numpy as np
import matplotlib.pyplot as plt


# helper
"""
def scaling(x, c):

    x = np.minimum(x, 1 / np.exp(1))
    temp = np.array([y * np.log(y) if y > 0 else 0 for y in x])
    temp /= ((1/c) * np.log((1/c)))
    temp[temp > 2] = 2
    out = 1 - np.abs((1 - temp))

    return out
"""

def trunc_linear(x, num_classes):

    B = 1 / num_classes

    index = (x <= 2 * B)
    output = np.zeros_like(x)
    output[index] = (B - np.abs(B - x[index])) / B

    return output


def trunc_entropy(x, num_classes):

    B = 1 / num_classes

    index = (x <= 2 * B)
    output = np.zeros_like(x)
    output[index] = (B - np.abs(B - x[index]))
    output = np.array([y * np.log(y) if y > 0 else 0 for y in output])
    output /= (B * np.log(B))

    return output


def scale_all(x, num_classes, entropy=0):

    B = 1 / num_classes

    index = (x > B)
    output = x.copy()
    output[index] = B * (output[index] - B) / (1 - B) + B

    return trunc_entropy(output, num_classes) if entropy else trunc_linear(output, num_classes)


""" Main function HERE """
def main():

    x = np.linspace(start=0, stop=1, num=200)
    plt.plot(x, scale_all(x, 3))
    plt.plot(x, scale_all(x, 5))
    plt.plot(x, scale_all(x, 10))

    fig, axarr = plt.subplots(ncols=4, sharey=True, figsize=(16, 4))

    plt.sca(axarr[0])
    plt.plot(x, trunc_linear(x, 3))
    plt.plot(x, trunc_entropy(x, 3))
    plt.plot(x, scale_all(x, 3))
    plt.plot(x, scale_all(x, 3, entropy=1))
    plt.legend(labels=['trunc-lin', 'trunc-ent', 'scale-lin', 'scale-ent'])
    plt.ylim(0, 1.1)
    plt.xlim(0, 1)
    plt.vlines((1 / 3), 0, 1, 'g', 'dashed')
    plt.text((1 / 3), 1.05, 'balanced', c='g')
    plt.title('D = 3')

    plt.sca(axarr[1])
    plt.plot(x, trunc_linear(x, 5))
    plt.plot(x, trunc_entropy(x, 5))
    plt.plot(x, scale_all(x, 5))
    plt.plot(x, scale_all(x, 5, entropy=1))
    plt.legend(labels=['trunc-lin', 'trunc-ent', 'scale-lin', 'scale-ent'])
    plt.ylim(0, 1.1)
    plt.xlim(0, 1)
    plt.vlines((1 / 5), 0, 1, 'g', 'dashed')
    plt.text((1 / 5), 1.05, 'balanced', c='g')
    plt.title('D = 5')

    plt.sca(axarr[2])
    plt.plot(x, trunc_linear(x, 10))
    plt.plot(x, trunc_entropy(x, 10))
    plt.plot(x, scale_all(x, 10))
    plt.plot(x, scale_all(x, 10, entropy=1))
    plt.legend(labels=['trunc-lin', 'trunc-ent', 'scale-lin', 'scale-ent'])
    plt.ylim(0, 1.1)
    plt.xlim(0, 1)
    plt.vlines((1 / 10), 0, 1, 'g', 'dashed')
    plt.text((1 / 10), 1.05, 'balanced', c='g')
    plt.title('D = 10')

    plt.sca(axarr[3])
    plt.plot(x, trunc_linear(x, 15))
    plt.plot(x, trunc_entropy(x, 15))
    plt.plot(x, scale_all(x, 15))
    plt.plot(x, scale_all(x, 15, entropy=1))
    plt.legend(labels=['trunc-lin', 'trunc-ent', 'scale-lin', 'scale-ent'])
    plt.ylim(0, 1.1)
    plt.xlim(0, 1)
    plt.vlines((1 / 15), 0, 1, 'g', 'dashed')
    plt.text((1 / 15), 1.05, 'balanced', c='g')
    plt.title('D = 15')

    plt.savefig('scaling.png')
    plt.show()


if __name__ == "__main__":
    main()
