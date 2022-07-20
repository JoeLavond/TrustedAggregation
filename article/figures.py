# packages
import numpy as np
import matplotlib.pyplot as plt


# helper
def scaling(x, c):

    temp = np.array([y * np.log(y) if y > 0 else 0 for y in x])
    temp /= ((1/c) * np.log((1/c)))
    temp[temp > 2] = 2
    out = 1 - (1 - temp) ** 2

    print(out)
    return out


def old_scaling(x, c):

    temp = np.array([y * np.log(y) if y > 0 else 0 for y in x])
    temp *= -c
    temp /= np.log(c)
    out = 1 - np.abs(1 - temp)
    print(out)

    return out


""" Main function HERE """
def main():

    x = np.linspace(start=0, stop=1, num=200)

    # scaling coefs
    c2 = scaling(x, c=2)
    c5 = scaling(x, c=5)
    c10 = scaling(x, c=10)
    c15 = scaling(x, c=15)

    fig, axarr = plt.subplots(ncols=4, sharey=True, figsize=(16, 4))

    plt.sca(axarr[0])
    plt.plot(x, c2)
    plt.ylim(0, 1.1)
    plt.vlines((1 / 2), 0, 1, 'g', 'dashed')
    plt.text((1 / 2), 1.05, 'balanced', c='g')
    plt.title('D = 2')

    plt.sca(axarr[1])
    plt.plot(x, c5)
    plt.ylim(0, 1.1)
    plt.vlines((1 / 5), 0, 1, 'g', 'dashed')
    plt.text((1 / 5), 1.05, 'balanced', c='g')
    plt.title('D = 5')

    plt.sca(axarr[2])
    plt.plot(x, c10)
    plt.ylim(0, 1.1)
    plt.vlines((1 / 10), 0, 1, 'g', 'dashed')
    plt.text((1 / 10), 1.05, 'balanced', c='g')
    plt.title('D = 10')

    plt.sca(axarr[3])
    plt.plot(x, c15)
    plt.ylim(0, 1.1)
    plt.vlines((1 / 15), 0, 1, 'g', 'dashed')
    plt.text((1 / 15), 1.05, 'balanced', c='g')
    plt.title('D = 15')

    plt.savefig('scaling.png')

    """ Previously """
    old_c2 = old_scaling(x, c=2)
    old_c5 = old_scaling(x, c=5)
    old_c10 = old_scaling(x, c=10)
    old_c15 = old_scaling(x, c=15)

    fig, axarr = plt.subplots(ncols=4, sharey=True, figsize=(16, 4))

    plt.sca(axarr[0])
    plt.plot(x, old_c2)
    plt.ylim(0, 1.1)
    plt.vlines((1 / 2), 0, 1, 'g', 'dashed')
    plt.text((1 / 2), 1.05, 'balanced', c='g')
    plt.title('D = 2')

    plt.sca(axarr[1])
    plt.plot(x, old_c5)
    plt.ylim(0, 1.1)
    plt.vlines((1 / 5), 0, 1, 'g', 'dashed')
    plt.text((1 / 5), 1.05, 'balanced', c='g')
    plt.title('D = 5')

    plt.sca(axarr[2])
    plt.plot(x, old_c10)
    plt.ylim(0, 1.1)
    plt.vlines((1 / 10), 0, 1, 'g', 'dashed')
    plt.text((1 / 10), 1.05, 'balanced', c='g')
    plt.title('D = 10')

    plt.sca(axarr[3])
    plt.plot(x, old_c15)
    plt.ylim(0, 1.1)
    plt.vlines((1 / 15), 0, 1, 'g', 'dashed')
    plt.text((1 / 15), 1.05, 'balanced', c='g')
    plt.title('D = 15')

    plt.savefig('old_scaling.png')


if __name__ == "__main__":
    main()
