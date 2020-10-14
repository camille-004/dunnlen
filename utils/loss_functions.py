import numpy as np


class LossFunction:
    def compute(self, y, y_hat):
        """Computes the loss function
        """
        raise NotImplementedError

    def compute_grad(self, y, y_hat):
        """Computes the gradient of the loss function
        """
        raise NotImplementedError


class MSE(LossFunction):
    def compute(self, y, y_hat):
        return np.mean(np.power(y - y_hat, 2))

    def compute_grad(self, y, y_hat):
        return 2 * (y_hat - y) / y.size
