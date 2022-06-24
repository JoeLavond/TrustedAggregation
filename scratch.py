import numpy as np
import torch
from torch.distributions.dirichlet import Dirichlet

import matplotlib.pyplot as plt
import seaborn as sns

def shannon(p):
    return -1 * sum([x * np.log(x) for x in p])

def simpson(p):
    return -1 * np.log(sum([x ** 2 for x in p]))

def berger(p):
    return -1 * np.log(max(p))


def main():

    # setup
    N = 1000
    c = 10
    alpha = (0.1, 0.5, 1, 5, 10, 100, 1000, 10000)

    for a in alpha:
        param = torch.full_like(torch.zeros(c), fill_value=a)
        dist = Dirichlet(param)

        temp_shannon = []
        temp_berger = []
        temp_simpson = []
        for n in range(N):
            sample = dist.sample().cpu().numpy()

            temp_shannon.append(shannon(sample))
            temp_berger.append(berger(sample))
            temp_simpson.append(simpson(sample))

        # visualize
        temp_shannon = [x / np.log(c) for x in temp_shannon]
        temp_berger = [x / np.log(c) for x in temp_berger]
        temp_simpson = [x / np.log(c) for x in temp_simpson]

        plt.figure()

        sns.kdeplot(temp_shannon)
        sns.kdeplot(temp_berger)
        sns.kdeplot(temp_simpson)

        plt.legend(labels=['shannon', 'berger', 'simpson'])
        plt.show()
        plt.close()




if __name__ == "__main__":
    main()
