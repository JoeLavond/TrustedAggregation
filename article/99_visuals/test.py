import os

def main():

    path = '/home/joe/03_federated/centralized/alpha10000--alpha_val10000'
    print([f.path for f in os.scandir(path) if re.search('', f.path)])


if __name__ == "__main__":
    main()
