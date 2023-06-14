from scipy.stats import multivariate_normal
from typing import List
from dataclasses import dataclass


@dataclass
class GaussianMixture:
    def __init__(self, means: List[float], covariances: List[float], weights: List = None):
        self.means = means
        self.covariances = covariances
        self.weights = weights if weights else [1/len(means)]*len(means)

        self.gaussians = [multivariate_normal(mean, covariance) for mean, covariance in zip(means, covariances)]

    def pdf(self, x):
        return sum([weight * gaussian.pdf(x) for weight, gaussian in zip(self.weights, self.gaussians)])
