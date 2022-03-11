import numpy as np
from sklearn import datasets
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

X, y = datasets.make_blobs(n_features=2, centers=2, random_state=666)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=666)

from perceptron_practice import Perceptron
perc = Perceptron(learning_rate=0.01, epochs=50)
perc.fit(X_train, y_train)
pred = perc.predict(X_test)

accuracy = np.sum(pred == y_test) / len(y_test)
print(accuracy)

# plt.figure(figsize=(12,8))
# plt.scatter(X[:,0], X[:,1], c=y, marker='o', alpha=0.5)
# plt.plot(X[:,0], X[:,1], c=y, marker='o', alpha=0.5)
# plt.show()

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
plt.scatter(X_train[:, 0], X_train[:, 1], marker="o", c=y_train)

x0_1 = np.amin(X_train[:, 0])
x0_2 = np.amax(X_train[:, 0])

x1_1 = (-perc.weights[0] * x0_1 - perc.bias) / perc.weights[1]
x1_2 = (-perc.weights[0] * x0_2 - perc.bias) / perc.weights[1]

ax.plot([x0_1, x0_2], [x1_1, x1_2], "k")

ymin = np.amin(X_train[:, 1])
ymax = np.amax(X_train[:, 1])
ax.set_ylim([ymin - 3, ymax + 3])

plt.show()

