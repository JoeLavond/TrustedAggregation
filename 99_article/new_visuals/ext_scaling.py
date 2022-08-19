# packages
import numpy as np
import matplotlib.pyplot as plt

# source
import local_utils as lu

""" Main function HERE """
def main():

    num_classes = 3
    x = np.linspace(start=0, stop=1, num=200)  # class proportions

    # create visual for k-way classification problem
    plt.figure()
    plt.plot(x, lu.quadratic_scaling(x, num_classes))
    plt.ylim(0, 1.1)
    plt.ylabel('Scaling')
    plt.xlim(0, 1)
    plt.xlabel('Class Frequency')
    plt.vlines((1 / num_classes), 0, 1, 'g', 'dashed')
    plt.text((1 / num_classes), 1.033, f'balanced for D = {num_classes}', c='g')

    plt.savefig('visuals/scaling.png')
    plt.show()


if __name__ == "__main__":
    main()
