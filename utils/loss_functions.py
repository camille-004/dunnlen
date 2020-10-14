import numpy as np


class LossFunction:
    def compute(self, *args, **kwargs):
        """Computes the loss function
        """
        raise NotImplementedError

    def compute_grad(self, *args, **kwargs):
        """Computes the gradient of the loss function
        """
        raise NotImplementedError
