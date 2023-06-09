# For installation guide, follow readme on: https://github.com/BGU-CS-VIL/dpmmpython
from julia.api import Julia
from dpmmpython.dpmmwrapper import DPMMPython
from dpmmpython.priors import niw
import numpy as np
jl = Julia(compiled_modules=False)


def main():
    data, gt = DPMMPython.generate_gaussian_data(10000, 2, 10, 100.0)
    prior = niw(1, np.zeros(2), 4, np.eye(2))
    labels, _, results = DPMMPython.fit(data, 100, prior=prior, verbose=True, gt=gt, gpu=False)


if __name__ == '__main__':
    main()
