import numpy as np


class LossFunction:
    def compute(self, *args, **kwargs):
        raise NotImplementedError

    def compute_grad(self, *args, **kwargs):
        raise NotImplementedError
