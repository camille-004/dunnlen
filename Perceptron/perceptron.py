import numpy as np

from neural_network import NeuralNetwork
from utils.activation_fcns import heaviside_step


class Perceptron(NeuralNetwork):
    """
    A class to implement a single-layer perceptron
    """
    def __init__(self, epochs, learning_rate=0.01, threshold=0):
        super().__init__(epochs)
        self.learning_rate = learning_rate
        self.threshold = threshold

        # Class attributes
        self.weights = None
        self.bias = None

    def activation(self, vec, threshold) -> object:
        """
        Map an input to a single binary value

        :param vec: real-valued vector
        :param threshold: where neuron takes activated value
        :return: single binary value
        """
        return heaviside_step(vec, threshold)

    def fit(self, features_train, labels_train):
        """
        Fit the perceptron model

        :param features_train: training samples
        :param labels_train: training labels
        :return: new weights and bias
        """
        # Initialize weights and bias
        self.weights = np.zeros(features_train.shape[1])
        self.bias = 0

        # Map labels to binary values
        pred_ = np.array([
            1 if i > self.threshold else 0 for i in labels_train
        ])

        for _ in range(self.epochs):
            for idx, x_i in enumerate(features_train):
                result = np.dot(x_i, self.weights) + self.bias
                predicted = self.activation(result, self.threshold)

                # Update rule based on LR, original predictions
                # and new predictions
                update_rule = self.learning_rate * (pred_[idx] - predicted)

                # Use update rule to update weights and bias
                self.weights += update_rule * x_i
                self.bias += update_rule

    def predict(self, vec):
        """
        Create predicted array from fit() or activation function

        :param vec: real-valued vector
        :return: NumPy array of predictions
        """
        # Linear output
        result = np.dot(vec, self.weights) + self.bias
        pred = self.activation(result, self.threshold)
        return pred

    def get_metrics(self, labels_true, labels_predicted):
        """
        Return model evaluation metrics

        :param labels_true: array containing actual labels
        :param labels_predicted: array containing predicted labels
        :return: accuracy, precision, recall, and f1 scores
        """
        super().get_metrics(labels_true, labels_predicted)

    def log_metrics(self, labels_true, labels_predicted):
        """
        Print model evaluation metrics

        :param labels_true: array containing actual labels
        :param labels_predicted: array containing predicted labels
        """
        super().log_metrics(labels_true, labels_predicted)
