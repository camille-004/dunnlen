import numpy as np


class Initializer:
    def create_params(self, dim, *args, **kwargs):
        """
        Initializes the parameters of a neural network according to function

        Parameters
        ----------
        dim : the dimensions of the tensor to initialize

        Returns
        -------
        the initialized values for the tensor
        """
        raise NotImplementedError


class Norm(Initializer):
    def create_params(self, dim: tuple, mean=0.0, std=1.0):
        return np.random.normal(loc=mean, scale=std, size=dim)
