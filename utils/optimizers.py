import numpy as np
from layers import Tensor


class Opt:
    """Class that implements an optimizer"""

    def __init__(self, params: list, lr=0.001) -> object:
        self.lr = lr
        self.cache = {}
        self.params = params

    def retrieve_cache(self, index):
        return self.cache[index]

    def into_cache(self, index, res):
        self.cache[index] = res

    def update(self):
        raise NotImplementedError

    def reset_grad(self):
        for param in self.params:
            param.grads = 0.


class SGD(Opt):
    """Stochastic gradient descent"""

    def __init__(self):
        super().__init__()

    def update(self):
        for param in self.params:
            param.data = param.data - self.lr * param.grads.mean(axis=0)

