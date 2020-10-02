import numpy as np


class Initializer:
    def create_params(self, dim, *args, **kwargs):
        raise NotImplementedError


class Norm(Initializer):
    def create_params(self, dim: tuple, mean=0.0, std=1.0):
        return np.random.normal(loc=mean, scale=std, size=dim)
