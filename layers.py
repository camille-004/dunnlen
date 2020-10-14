import numpy as np


class Tensor:
    """Class that implements a tensor"""

    def __init__(self, dim: tuple):
        self.data = np.ndarray(dim, np.float32)
        self.grads = np.ndarray(dim, np.float32)


class Layer:
    def __init__(self):
        self.type = 'layer'

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def backward(self, *args, **kwargs):
        raise NotImplementedError

    @staticmethod
    def get_params():
        return []


class Dense(Layer):
    """Implements an affine transformation dense layer"""

    def __init__(self, dims: tuple):
        super().__init__()
        self.W = Tensor(dims)
        self.b = Tensor((1, dims[1]))
        self.input = []
        self.type = 'dense'

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

    def backward(self, upstream_gradient, lr=0.001):
        """

        Parameters
        ----------
        upstream_gradient : Gradients of loss function w.r.t. layer output
        lr : learning rate

        Returns
        -------
        Partial derivative of loss w.r.t. layer input and parameters, will be
        passed to previous layer
        """
        inp_error = np.dot(upstream_gradient, self.W.data.T)
        print(inp_error)
        W_error = np.dot(self.input.T, upstream_gradient)

        self.W += lr * W_error
        self.b += lr * upstream_gradient

        return inp_error

    def get_params(self):
        return [self.W, self.b]


class Activation(Layer):
    """Implements an activation layer"""

    def __init__(self, activation, activation_deriv):
        super().__init__()
        self.activation = activation
        self.activation_deriv = activation_deriv
        self.type = 'activation'

    def forward(self, feature_matrix):
        """Forward propagation for activation layer

        Returns
        -------
        activation function of input data
        """
        self.feature_matrix = feature_matrix
        self.result = self.activation(self.feature_matrix)
        return self.result

    def backward(self, error):
        """Backward propagation for activation layer

        Parameters
        ----------
        error : dE / dY

        Returns
        -------
        input error dE / dX given output error dE / dY
        """
        return self.activation_deriv(self.feature_matrix) * error
