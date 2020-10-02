import numpy as np
from initializer import Initializer


class Tensor:
    def __init__(self, dim: tuple, init: Initializer):
        self.data = init.create_params(dim)
        self.grads = np.ndarray(dim, np.float32)


class Layer:
    def __init__(self):
        pass

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def backward(self, *args, **kwargs):
        raise NotImplementedError

    @staticmethod
    def get_params():
        return []


class Dense(Layer):
    def __init__(self, dims: tuple, w_init: Initializer, b_init: Initializer):
        super().__init__()
        self.W = Tensor(w_init.create_params(dims))
        self.b = Tensor(b_init.create_params(1, dims[1]))
        self.input = []

    def forward(self, feature_matrix):
        """Affine transformation, and saves input, which is needed to compute
        gradients of weights in backward()

        Parameters
        ----------
        feature_matrix
            Layer input

        Returns
        -------
        Transformation done by layer
        """
        res = np.dot(feature_matrix, self.W.data) + self.b.data
        self.input = feature_matrix
        return res

    def backward(self, upstream_gradient):
        """

        Parameters
        ----------
        upstream_gradient
            Gradients of loss function w.r.t. layer output

        Returns
        -------
        Partial derivative of loss w.r.t. layer input and parameters, will be
        passed to previous layer
        """
        self.W.grads += np.dot(self.input.T, upstream_gradient)
        self.b.grads += np.sum(upstream_gradient, axis=0, keepdims=True)
        grad_input = np.dot(upstream_gradient, self.W.data.T)
        return grad_input

    def get_params(self):
        return [self.W, self.b]
