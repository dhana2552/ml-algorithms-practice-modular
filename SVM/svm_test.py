import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.metrics import accuracy_score

from svm_practice import SVM

X, y = datasets.make_blobs(n_samples=100, n_features=2, centers=2, random_state=42)
y = np.where(y <= 0, -1, 1)

clf = SVM()
clf.fit(X, y)

print(clf.weight, clf.bias)

def get_hyperplane_value(x, w, b, offset):
    return (-w[0] * x + b + offset) / w[1]

fig = plt.figure()
ax = fig.add_subplot(111)
plt.scatter(X[:, 0], X[:, 1], marker='o', c=y)

x0_1 = np.amin(X[:, 0])
x0_2 = np.amax(X[:, 0])

x1_1 = get_hyperplane_value(x0_1, clf.weight, clf.bias, 0)
x1_2 = get_hyperplane_value(x0_2, clf.weight, clf.bias, 0)

x1_1_m = get_hyperplane_value(x0_1, clf.weight, clf.bias, -1)
x1_2_m = get_hyperplane_value(x0_2, clf.weight, clf.bias, -1)

x1_1_p = get_hyperplane_value(x0_1, clf.weight, clf.bias, 1)
x1_2_p = get_hyperplane_value(x0_2, clf.weight, clf.bias, 1)

ax.plot([x0_1, x0_2], [x1_1, x1_2], "k")
ax.plot([x0_1, x0_2], [x1_1_m, x1_2_m], "y--")
ax.plot([x0_1, x0_2], [x1_1_p, x1_2_p], "y--")

x1_min = np.amin(X[:, 1])
x1_max = np.amax(X[:, 1])
ax.set_ylim([x1_min - 3, x1_max + 3])

plt.show()

predictions = clf.predict(X)
accuracy_score = np.sum(predictions == y)/len(y)

print(accuracy_score)