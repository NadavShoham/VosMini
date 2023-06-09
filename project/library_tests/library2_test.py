from julia.api import Julia
from dpmmpythonStreaming.dpmmwrapper import DPMMPython
from dpmmpythonStreaming.priors import niw
import numpy as np
jl = Julia(compiled_modules=False)


def main():
    data, gt = DPMMPython.generate_gaussian_data(10000, 2, 10, 100.0)
    batch1 = data[:, 0:5000]
    batch2 = data[:, 5000:]
    prior = DPMMPython.create_prior(2, 0, 1, 1, 1)
    model = DPMMPython.fit_init(batch1, 100.0, prior=prior, verbose=True, burnout=5, gt=None, epsilon=0.0000001)
    labels = DPMMPython.get_labels(model)
    print(labels)
    model = DPMMPython.fit_partial(model, 1, 2, batch2)
    labels = DPMMPython.get_labels(model)
    print(labels)


if __name__ == '__main__':
    main()
