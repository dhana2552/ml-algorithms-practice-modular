import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt

X, y = datasets.make_regression(n_samples=100, n_features=1, noise=20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# plt.figure()
# plt.scatter(X, y, color='red', s=30)
# plt.show()

from linear_regression_practice import LinearRegression
reg = LinearRegression(0.01)
reg.fit(X_train, y_train)
predictions = reg.predict(X_test)

mse = np.mean((y_test - predictions) ** 2)
print(mse)

plt.figure()
plt.scatter(X, y, color='red', s=30)
plt.plot(X, reg.predict(X), color='blue', linewidth=3)
plt.show()