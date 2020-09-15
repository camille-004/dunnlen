from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

from .perceptron import Perceptron

X, y = datasets.make_blobs(n_samples=150, n_features=2, centers=2, cluster_std=1.05, random_state=2)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

p = Perceptron(1000)
p.fit(X_train, y_train)
pred = p.predict(X_test)
p.log_metrics(y_test, pred)

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
plt.scatter(X_train[:, 0], X_train[:, 1], marker='x', c=y_train)

x1 = np.amin(X_train[:, 0])
x2 = np.amax(X_train[:, 0])

y1 = (-p.weights[0] * x1 - p.bias) / p.weights[1]
y2 = (-p.weights[0] * x2 - p.bias) / p.weights[1]

ax.plot([x1, x2], [y1, y2], 'k')
ax.set_title('Decision Boundary on Test Dataset')
plt.show()
