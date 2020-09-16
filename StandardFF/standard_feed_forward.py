import numpy as np

from neural_network import NeuralNetwork
from utils.activation_fcns import sigmoid, reLu, sigmoid_deriv, reLu_deriv

'''Testing'''
import pandas as pd
from sklearn.model_selection import train_test_split


class StandardFF(NeuralNetwork):
    """
    A class to implement a standard feed-forward network.
    Useful for any binary classification problem.

    TODO:
    1. Finish fit()
    2. Implement loss functions --> compute loss in fit()
    3. Implement predict()
    4. Implement metrics
    5. Test
    """

    def __init__(
            self, epochs, learning_rate, layer_dimensions, activation_layers):
        super().__init__(epochs, learning_rate)
        self.layer_dimensions = layer_dimensions
        self.activation_seq = activation_layers

        self.L = len(self.layer_dimensions)
        self.n_examples = None
        self.params = {}
        self.losses = []

    def activation(self, act_func, vec):
        """
        Determine activation function to be used on a layer

        :param act_func: activation function to use
        :param vec: real-valued vector
        :return: result of activation function
        """
        if act_func == 'sigmoid':
            return sigmoid(vec)
        elif act_func == 'reLu':
            return reLu(vec)

    @staticmethod
    def activation_deriv(act_func, vec):
        """
        Calculate a vector from the derivative of the given activation function
        Products of layer derivatives computed as loss function derivative

        :param act_func: activation function to differentiate
        :param vec: real-valued vector
        :return: result of differentiated activation function
        :param d_act: parameter for reLu backward calculation
        """
        assert act_func in ['sigmoid', 'reLu']
        if act_func == 'sigmoid':
            return sigmoid_deriv(vec)
        elif act_func == 'reLu':
            return reLu_deriv(vec)

    def init_weights_zeros(self, layer_dimensions):
        """
        Initialize the weights with zeros

        :param layer_dimensions: A list containing the number of nodes in each
        layer
        :return: a dictionary containing the initialized weights and biases for
        each layer
        """
        for i in range(1, len(layer_dimensions)):
            self.params['weight_' + i] = np.zeros(
                layer_dimensions[i], layer_dimensions[i - 1] * 10)
            self.params['bias_' + i] = np.zeros((layer_dimensions[i], 1))

    def init_weights_rand(self, layer_dimensions):
        """
        Initialize the weights by randomly from dimensions of each layer

        :param layer_dimensions: A list containing the number of nodes in each
        layer
        """
        for i in range(1, len(layer_dimensions)):
            self.params['weight_' + i] = np.random.randn(
                layer_dimensions[i], layer_dimensions[i - 1] * 10
            )
            self.params['bias_' + i] = np.zeros((layer_dimensions[i], 1))

    def init_weights_he(self, layer_dimensions):
        """
        Initialize the weights by method from He et al. (2015)

        :param layer_dimensions: A list containing the number of nodes in each
        layer
        """
        for i in range(1, len(layer_dimensions)):
            self.params['weight_' + str(i)] = np.random.randn(
                layer_dimensions[i], layer_dimensions[i - 1]) * np.sqrt(
                2 / layer_dimensions[i - 1])
            self.params['bias_' + str(i)] = np.zeros((layer_dimensions[i], 1))

    def forward_propagation(self, features_train):
        """
        Implements forward-propagation algorithm

        :param features_train: input dataset for element-wise weighted sum
        :return: weighted sum and activation results stored in a cache array
        """
        cache = {}

        # For weighted sum
        activation_Z = np.array(features_train.T)

        # Exclude the final layer
        for layer in range(self.L - 1):
            Z = np.dot(self.params['weight_' + str(layer + 1)], activation_Z) + self.params['bias_' + str(layer + 1)]
            activation_Z = self.activation(self.activation_seq[layer], Z)
            cache['act_' + str(layer + 1)] = activation_Z
            cache['weight_' + str(layer + 1)] = self.params[
                'weight_' + str(layer + 1)]
            cache['Z_' + str(layer + 1)] = Z

        # For final output layer, for SoftMax?
        Z = self.params[
                'weight_' + str(self.L)].dot(activation_Z) + self.params[
                'bias_' + str(self.L)]
        y_hat = self.activation(self.activation_seq[self.L - 1], Z)
        cache['act_' + str(self.L)] = activation_Z
        cache['weight_' + str(self.L)] = self.params[
            'weight_' + str(self.L)]
        cache['Z_' + str(self.L)] = Z

        return y_hat, cache

    def backward_propagation(self, features_train, labels_train, cache):
        """
        Implements backward-propagation algorithm

        :param features_train: back-propagation parameter
        :param labels_train:
        :param cache:
        :return: dictionary of gradients
        """
        gradients = {}
        cache['act_0'] = features_train.T
        y_hat = cache['act_' + str(self.L)]
        dy_hat = -np.divide(
            labels_train, y_hat) + np.divide(
            1 - labels_train, 1 - y_hat)

        # Derivative of weighted sum
        dZ = dy_hat * self.activation_deriv(
            cache['Z_' + str(self.L)], self.activation_seq[self.L])
        d_weight = dZ.dot(
            cache['act_' + str(self.L - 1)].T) / features_train.shape[0]
        d_bias = np.sum(dZ, axis=1, keepdims=True)
        dA_prev = cache['weight_' + str(self.L)].T.dot(dZ)

        gradients['weight_' + str(self.L)] = d_weight
        gradients['bias_' + str(self.L)] = d_bias

        for layer in range(self.L - 1, 0, -1):
            # Calculate gradients for backward layers
            dZ = dA_prev * self.activation_deriv(
                cache['Z_' + str(layer)], self.activation_seq[layer])
            dW = 1 / features_train.shape[0] * dZ.dot(
                cache['act_' + str(layer - 1)].T)
            db = 1 / features_train.shape[0] * np.sum(
                dZ, axis=1, keepdims=True)
            if layer > 1:
                dA_prev = cache['weight_' + str(layer)].T.dot(dZ)

            # Update gradients container
            gradients['weight_' + str(layer)] = dW
            gradients['bias_' + str(layer)] = db

        return gradients

    def loss(self, activation_result, labels_train):
        """
        Compute binary cross-entropy loss, AKA log loss

        :param activation_result: Parameter for cross-entropy
        :param labels_train: Get shape for calculating cross-entropy
        :return: Result of cost function for each epoch, one-dimensional
        aspects removed
        """
        return np.squeeze(
            -(labels_train.dot(
                np.log(
                    activation_result.T
                )
            ) + (1 - labels_train).dot(np.log(
                1 - activation_result.T))) / self.n_examples)

    def fit(self, features_train, labels_train, init_method):
        """
        Fit our model by running f-prop, loss, and b-prop

        :param features_train: Training feature to fit to
        :param labels_train: Training labels to fit to
        :param init_method: Method of initializing weights and biases
        """
        # Weight needs number of features
        n_features = features_train.shape[1]
        self.layer_dimensions.insert(0, n_features)
        self.n_examples = features_train.shape[0]

        # Initialize parameters
        if init_method == 'zeros':
            self.init_weights_zeros(self.layer_dimensions)
        elif init_method == 'rand':
            self.init_weights_rand(self.layer_dimensions)
        elif init_method == 'he':
            self.init_weights_he(self.layer_dimensions)

        for epoch in range(self.epochs):
            # Run forward propagation
            A, cache = self.forward_propagation(features_train)

            # Calculate cost
            cost = self.loss(A.T, labels_train)

            # Run backward propagation
            gradients = self.backward_propagation(
                features_train, labels_train)

            # Update params
            for j in range(1, self.L + 1):
                self.params[
                    'weight_' + str(j)
                    ] = self.params[
                            'weight_' + str(j)
                            ] - self.learning_rate * gradients[
                            'd_weight' + str(j)]
                self.params[
                    'bias_' + self.params(j)
                    ] = ['bias_' + str(j)] - self.learning_rate

            self.losses.append(cost)

    def predict(self, features_test):
        """
        Get predicted output from input into our fitted model

        :param features_test: input dataset from which to generate predictions
        :return: NumPy array of predictions
        """
        A, cache = self.forward_propagation(features_test)
        res = np.zeros((1, features_test.shape[0]))

        for i in range(A.shape[1]):
            if A[0, i] > 0.5:
                res[0, i] = 1

        return res


if __name__ == '__main__':
    n = 100

    data = []
    for i in range(n):
        temp = {}
        temp.update({'temperature': np.random.normal(14, 3)})
        temp.update({'moisture': np.random.normal(96, 2)})

        label = 0
        if temp['temperature'] < 10 or temp['temperature'] > 18:
            label = 1
        elif temp['moisture'] < 94 or temp['moisture'] > 98:
            label = 1
        temp.update({'label': label})

        data.append(temp)

    df = pd.DataFrame(data=data)
    df.head()

    X = df[['temperature', 'moisture']]
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
    # print("X_train shape: ", X_train.shape)
    # print("X_test shape: ", X_test.shape)
    # print("y_train shape: ", y_train.shape)
    # print("y_test shape: ", y_test.shape)

    layer_dimensions = [40, 20, 10, 5]
    activation_layers = ['sigmoid', 'reLu', 'reLu', 'sigmoid']

    nn = StandardFF(200, 0.5, layer_dimensions, activation_layers)
    nn.fit(X_train, y_train, 'he')
