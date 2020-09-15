from utils.metrics import *


class NeuralNetwork:
    """
    A class to implement a neural network
    """

    def __init__(self, epochs, learning_rate=0.01, reg_term=0, layer_dimensions=3):
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.reg_term = reg_term
        self.layer_dimensions = layer_dimensions

    def activation(self, vec):
        pass

    def fit(self, features_train, labels_train):
        pass

    def predict(self, vec):
        pass

    def get_metrics(self, labels_true, labels_predicted):
        """
        Return model evaluation metrics

        :param labels_true: array containing actual labels
        :param labels_predicted: array containing predicted labels
        :return: accuracy, precision, recall, and f1 scores
        """
        return_metrics(labels_true, labels_predicted)

    def log_metrics(self, labels_true, labels_predicted):
        """
        Print model evaluation metrics

        :param labels_true: array containing actual labels
        :param labels_predicted: array containing predicted labels
        """
        print_metrics(labels_true, labels_predicted)
