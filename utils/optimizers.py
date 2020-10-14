import numpy as np
from layers import Tensor


class Opt:
    """Class that implements an optimizer"""

    def __init__(self, params: list):
        self.cache = {}
        self.params = params

    def retrieve_cache(self, index):
        """Retrieves result of optimizer from cache"""
        return self.cache[index]

    def into_cache(self, index, res):
        """Saves result of optimizer into cache"""
        self.cache[index] = res

    def update(self):
        """Updates the optimizer"""
        raise NotImplementedError

    def reset_grad(self):
        """Reset to zero gradient in each epoch and iteration"""
        for param in self.params:
            param.grads = 0.


class SGD(Opt):
    """Stochastic gradient descent"""

    def __init__(self, params, lr=0.001):
        super().__init__(params)
        self.lr = lr

    def update(self):
        for param in self.params:
            param.data = param.data - self.lr * param.grads.mean(axis=0)
