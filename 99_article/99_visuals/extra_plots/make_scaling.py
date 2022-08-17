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

def scaling(x, num_classes):
    c = 1 / np.exp(1)

    out = np.zeros_like(x)
    for i, y in enumerate(x):
        if y == 0:
            pass
        else:

            if y <= c:
                out[i] = y * np.log(y)
            else:
                out[i] = 2 * c * np.log(c) - y * np.log(y)

    out /= ((1 / num_classes) * np.log((1 / num_classes)))
    #temp[temp > 2] = 2
    out = 1 - np.abs((1 - out))

    return out



""" Main function HERE """
def main():

    x = np.linspace(start=0, stop=1, num=200)

    # scaling coefs
    abs_c3 = scaling(x, 3)
    abs_c5 = scaling(x, 5)
    abs_c10 = scaling(x, 10)
    abs_c15 = scaling(x, 15)

    fig, axarr = plt.subplots(ncols=4, sharey=True, figsize=(16, 4))

    plt.sca(axarr[0])
    plt.plot(x, abs_c3)
    plt.ylim(0, 1.1)
    plt.vlines((1 / 3), 0, 1, 'g', 'dashed')
    plt.text((1 / 3), 1.05, 'balanced', c='g')
    plt.title('D = 3')

    plt.sca(axarr[1])
    plt.plot(x, abs_c5)
    plt.ylim(0, 1.1)
    plt.vlines((1 / 5), 0, 1, 'g', 'dashed')
    plt.text((1 / 5), 1.05, 'balanced', c='g')
    plt.title('D = 5')

    plt.sca(axarr[2])
    plt.plot(x, abs_c10)
    plt.ylim(0, 1.1)
    plt.vlines((1 / 10), 0, 1, 'g', 'dashed')
    plt.text((1 / 10), 1.05, 'balanced', c='g')
    plt.title('D = 10')

    plt.sca(axarr[3])
    plt.plot(x, abs_c15)
    plt.ylim(0, 1.1)
    plt.vlines((1 / 15), 0, 1, 'g', 'dashed')
    plt.text((1 / 15), 1.05, 'balanced', c='g')
    plt.title('D = 15')

    plt.savefig('scaling.png')
    plt.show()


if __name__ == "__main__":
    main()
