from utils.get_data import GetData
from utils.optimizers import SGD
from utils.initializer import Norm


class Net:
    """Class that implements a neural network"""

    def __init__(self, initializer='norm'):
        # Create computation graph, define a sequential model
        self.comp_graph = []
        self.params = []
        if initializer == 'norm':
            self.initializer = Norm()

    def add(self, layer):
        """Adds a layer to the neural network

        Parameters
        ----------
        layer : layer to add
        """
        self.comp_graph.append(layer)
        self.params += layer.get_params()

    def initialize(self):
        """Initializes the parameters of the neural network"""
        for layer in self.comp_graph:
            if layer.type == 'dense':
                # Initialize Dense weights and bias
                W, b = layer.get_params()
                W.data = self.initializer.create_params(dim=(W.data.shape[0], W.data.shape[1]))
                b.data = self.initializer.create_params(dim=(1, b.data.shape[1]))

    def fit(self, data, target, epochs, loss, opt, batch_size=32):
        """
        Fits the neural network

        Parameters
        ----------
        data : X data
        target : y data
        batch_size : number of training samples for one iteration
        epochs : number of runs through the training data
        opt : optimizer to use
        loss : loss function to use

        Returns
        -------
        the loss history
        """
        history = []
        self.initialize()
        data = GetData(data, target, batch_size)
        iteration = 0
        for epoch in range(epochs):
            for features, labels in data:
                opt.reset_grad()
                for layer in self.comp_graph:
                    features = layer.forward(features)
                loss_result = loss.compute(features, labels)
                grad = loss.compute_grad(features, labels)
                print(grad.shape)
                for layer in reversed(self.comp_graph):
                    grad = layer.backward(grad)
                history += [loss_result]
                print(
                    f'Loss at epoch #{epoch},'
                    f'iteration #{iteration}: {history[-1]}')
                iteration += 1
                opt.update()

        return history

    def predict(self, data):
        """Generates data from model inference

        Parameters
        ----------
        data : predict the target for this data

        Returns
        -------
        predictions
        """
        for layer in self.comp_graph:
            data = layer.forward(data)
        return data
