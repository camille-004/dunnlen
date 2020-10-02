import numpy as np


class Opt:
    """Class that implements an optimizer"""

    def __init__(self, lr):
        self.lr = lr

    def forward(self, grads, w_b):
        step_vals = self.update(grads)
        grads = step_vals
        w_b += grads

    def update(self):
        raise NotImplementedError


class SGD(Opt):
    """Stochastic gradient descent"""

    def __init__(self, lr=0.001):
        super().__init__(lr)

    def update(self, grads):
        return -self.lr * grads


class Momentum(Opt):
    """Uses laws of motion to pass through local optima, increase speed of
    convergence"""

    def __init__(self, lr=0.001, gamma=0.9):
        super().__init__(lr)
        self.gamma = gamma
        self.accumulation = 0

    def update(self, grads):
        self.accumulation = self.gamma * self.accumulation + grads
        forward = -self.lr * self.accumulation
        return forward
