import numpy as np

class LogisticRegression:
    def __init__(self, lr=0.001, num_iter=1000):
        self.lr = lr
        self.num_iter = num_iter
        self.weights = None
        self.bias = None
        
    def _sigmoid(self, z):
        return 1/(1 + np.exp(-z))
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        for _ in range(self.num_iter):
            linear_model = np.dot(X, self.weights) + self.bias
            pred = self._sigmoid(linear_model)
            dw = np.dot(X.T, (pred - y)) / n_samples
            db = np.sum(pred - y) / n_samples
            self.weights -= self.lr * dw
            self.bias -= self.lr * db
            
    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        y_pred = self._sigmoid(linear_model)
        y_pred = [1 if i>0.5 else 0 for i in y_pred]
        return y_pred