import numpy as np

from neural_network import NeuralNetwork
from utils.activation_fcns import sigmoid, reLu, sigmoid_deriv, reLu_deriv


class StandardFF(NeuralNetwork):
    """
    A class to implement a standard feed-forward network.
    Recommended only for small-scale NNs.

    TODO:
    1. Finish fit()
    2. Implement loss functions --> compute loss in fit()
    3. Test
    """

    def __init__(self, epochs, layer_dimensions, activation_layers):
        super().__init__(epochs, layer_dimensions)
        self.layer_dimensions = layer_dimensions
        self.activation_layers = activation_layers

    def activation(self, act_func, vec):
        """
        Determine activation function to be used on a layer

        :param act_func: activation function to use
        :param vec: real-valued vector
        :return: result of activation function
        """
        assert act_func in ['sigmoid', 'reLu']
        if act_func == 'sigmoid':
            return sigmoid(vec)
        elif act_func == 'relu':
            return reLu(vec)

    @staticmethod
    def activation_deriv(act_func, vec, d_act=0):
        """
        Calculate a vector from the derivative of the given activation function

        :param act_func: activation function to differentiate
        :param vec: real-valued vector
        :return: result of differentiated activation function
        :param d_act: parameter for reLu backward calculation
        """
        assert act_func in ['sigmoid', 'reLu']
        if act_func == 'sigmoid':
            return sigmoid_deriv(vec)
        elif act_func == 'relu':
            return reLu_deriv(d_act, vec)

    @staticmethod
    def init_weights_zeros(layer_dimensions):
        """
        Initialize the weights with zeros

        :param layer_dimensions: A list containing the number of nodes in each
        layer
        :return: a dictionary containing the initialized weights and biases for
        each layer
        """
        # Store connective parameters in dictionary because it's dynamic
        params = {}
        for i in range(1, len(layer_dimensions) + 1):
            params['weight_' + i] = np.zeros(
                layer_dimensions[i], layer_dimensions[i - 1] * 10)
            params['bias_' + i] = np.zeros((layer_dimensions[i], 1))

        return params

    @staticmethod
    def init_weights_rand(layer_dimensions):
        """
        Initialize the weights by randomly from dimensions of each layer

        :param layer_dimensions: A list containing the number of nodes in each
        layer
        :return: a dictionary containing the initialized weights and biases for
        each layer
        """
        params = {}
        for i in range(1, len(layer_dimensions) + 1):
            params['weight_' + i] = np.random.randn(
                layer_dimensions[i], layer_dimensions[i - 1] * 10
            )
            params['bias_' + i] = np.zeros((layer_dimensions[i], 1))

        return params

    @staticmethod
    def init_weights_he(layer_dimensions):
        """
        Initialize the weights by method from He et al. (2015)

        :param layer_dimensions: A list containing the number of nodes in each
        layer
        :return: a dictionary containing the initialized weights and biases for
        each layer
        """
        params = {}
        for i in range(1, len(layer_dimensions) + 1):
            params['weight_' + str(i)] = np.random.randn(
                layer_dimensions[i], layer_dimensions[i - 1]) * np.sqrt(
                2 / layer_dimensions[i - 1])
            params['bias_' + str(i)] = np.zeros((layer_dimensions[i], 1))

        return params

    def forward_propagation(self, features_train, connection_params):
        """
        Implements forward-propagation algorithm

        :param features_train: input dataset for element-wise weighted sum
        :param connection_params: dictionary of weights and biases
        :return: weighted sum and activation results stored in a cache array
        """
        weights = [
            connection_params[
                k] for k in connection_params if 'weight' in connection_params]
        biases = [
            connection_params[
                k] for k in connection_params if 'bias' in connection_params]
        cache = {}

        for  i in range(len(weights)):
            # Calculate weighted sum
            z = np.dot(features_train, weights[i]) + biases[i]
            activate = self.activation(self.activation_layers[i],z)
            cache['z_' + str(i)] = z
            cache['act_' + str(i)] = activate

        return activate, cache

    def back_propagation(
            self, activations_result, labels_train, connection_params,
            forward_cache):
        """
        Implements backward-propagation algorithm

        :param activations_result: back-propagation parameter
        :param labels_train:
        :param connection_params:
        :param forward_cache:
        :return: dictionary of gradients
        """
        gradients = {}
        labels_train = labels_train.reshape(activations_result.shape)

        # Initialize backpropagation
        d_last_layer = (
                        (1 - labels_train) / (1 - activations_result)
                ) - (labels_train / activations_result)
        d_prev = d_last_layer

        for i in reversed(range(len(self.layer_dimensions))):
            curr_d_act = d_prev

            # Get current activation function, weight, weighted sum
            act_func = self.activation_layers[i]
            curr_weight = connection_params['weight_' + str(i)]
            curr_z = forward_cache['z_' + str(i)]
            prev_act = forward_cache['act_' + str(i - 1)]

            if act_func == 'sigmoid':
                d_curr_z = sigmoid_deriv(curr_d_act, curr_z)
            elif act_func == 'reLu':
                d_curr_z = reLu_deriv(curr_d_act, curr_z)
            d_weight = np.dot(d_curr_z, prev_act.T) / prev_act.shape[1]
            d_bias = np.sum(d_curr_z, axis=1, keepdims=True) / prev_act.shape[1]

            # Update layer
            d_prev = np.dot(d_curr_z, curr_weight.T)

            # Update gradients
            gradients['d_weight_' + str(i)] = d_weight
            gradients['d_bias_' + str(i)] = d_bias

        return gradients


    def fit(
            self, features_train, labels_train, init_method):
        """
        Fit Multi-Layer Perceptron

        :param features_train: Feature vectors for forward propagation
        :param labels_train: Label vectors for backward propagation
        :param init_method: Weight and bias initialization method
        :return: Updated and final parameters
        """
        assert init_method in ['zeros', 'rand', 'he']

        if init_method == 'zeros':
            params = self.init_weights_zeros(self.layer_dimensions)
        elif init_method == 'rand':
            params = self.init_weights_rand(self.layer_dimensions)
        elif init_method == 'he':
            params = self.init_weights_he(self.layer_dimensions)

        for i in range(self.epochs):
            activations_result, cache = self.forward_propagation(features_train, params)

            gradients = self.back_propagation(activations_result, labels_train, params, cache)

            # Update parameters
            for j in range(1, len(params) + 1):
                params[
                    'weight_' + str(j)
                    ] = params['weight_' + str(j)
                               ] - self.learning_rate * gradients[
                    'd_weight' + str(j)]
                params[
                    'bias_' + str(j)
                    ] = params['bias_' + str(j)
                               ] - self.learning_rate * gradients[
                    'd_ bias' + str(j)]

        return params
