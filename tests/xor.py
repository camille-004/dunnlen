import numpy as np

from layers import Dense, Activation
from utils.activation_fcns import tanh, tanh_deriv
from utils.loss_functions import MSE
from utils.optimizers import SGD
from net import Net

# Define training data
X_train = np.array([[[0, 0]], [[0, 1]], [[1, 0]], [[1, 1]]])
y_train = np.array([[[0]], [[1]], [[1]], [[0]]])

# Define and train neural network
nn = Net()
nn.add(Dense((2, 3)))
nn.add(Activation(activation=tanh, activation_deriv=tanh_deriv))
nn.add(Dense((3, 1)))
nn.add(Activation(activation=tanh, activation_deriv=tanh_deriv))

nn.fit(X_train, y_train, epochs=1000, loss=MSE(), opt=SGD(nn.params))

out = nn.predict(X_train)
print(out)