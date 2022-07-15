# packages
import numpy as np
import matplotlib.pyplot as plt


# helper
def scaling(x, c):

    temp = np.array([y * np.log(y) if y > 0 else 0 for y in x])
    temp /= ((1/c) * np.log((1/c)))
    temp[temp > 2] = 2

    return 1 - (1 - temp) ** 2


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
    plt.title('D = 2')

    plt.sca(axarr[1])
    plt.plot(x, c5)
    plt.title('D = 5')

    plt.sca(axarr[2])
    plt.plot(x, c10)
    plt.title('D = 10')

    plt.sca(axarr[3])
    plt.plot(x, c15)
    plt.title('D = 15')

    plt.savefig('scaling.png')


if __name__ == "__main__":
    main()
