# packages
import numpy as np
import matplotlib.pyplot as plt


# helper
def sq_scaling(x, c):

    temp = np.array([y * np.log(y) if y > 0 else 0 for y in x])
    temp /= ((1/c) * np.log((1/c)))
    temp[temp > 2] = 2
    out = 1 - (1 - temp) ** 2

    print(out)
    return out


def abs_scaling(x, c):

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
    sq_c2 = sq_scaling(x, c=2)
    sq_c5 = sq_scaling(x, c=5)
    sq_c10 = sq_scaling(x, c=10)
    sq_c15 = sq_scaling(x, c=15)

    fig, axarr = plt.subplots(ncols=4, sharey=True, figsize=(16, 4))

    plt.sca(axarr[0])
    plt.plot(x, sq_c2)
    plt.ylim(0, 1.1)
    plt.vlines((1 / 2), 0, 1, 'g', 'dashed')
    plt.text((1 / 2), 1.05, 'balanced', c='g')
    plt.title('D = 2')

    plt.sca(axarr[1])
    plt.plot(x, sq_c5)
    plt.ylim(0, 1.1)
    plt.vlines((1 / 5), 0, 1, 'g', 'dashed')
    plt.text((1 / 5), 1.05, 'balanced', c='g')
    plt.title('D = 5')

    plt.sca(axarr[2])
    plt.plot(x, sq_c10)
    plt.ylim(0, 1.1)
    plt.vlines((1 / 10), 0, 1, 'g', 'dashed')
    plt.text((1 / 10), 1.05, 'balanced', c='g')
    plt.title('D = 10')

    plt.sca(axarr[3])
    plt.plot(x, sq_c15)
    plt.ylim(0, 1.1)
    plt.vlines((1 / 15), 0, 1, 'g', 'dashed')
    plt.text((1 / 15), 1.05, 'balanced', c='g')
    plt.title('D = 15')

    plt.savefig('sq_scaling.png')


    # scaling coefs
    abs_c2 = abs_scaling(x, c=2)
    abs_c5 = abs_scaling(x, c=5)
    abs_c10 = abs_scaling(x, c=10)
    abs_c15 = abs_scaling(x, c=15)

    fig, axarr = plt.subplots(ncols=4, sharey=True, figsize=(16, 4))

    plt.sca(axarr[0])
    plt.plot(x, abs_c2)
    plt.ylim(0, 1.1)
    plt.vlines((1 / 2), 0, 1, 'g', 'dashed')
    plt.text((1 / 2), 1.05, 'balanced', c='g')
    plt.title('D = 2')

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

    plt.savefig('abs_scaling.png')

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
