import numpy as np

class Perceptron:
    def __init__(self, num_features, learning_rate=0.01, epochs=100) -> None:
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = np.zeros(num_features)
        self.bias = 0

    # using Heaviside function
    def activation(self, x):
        return 1 if x >= 0 else -1

    def predict(self, X):
        # f([data])
        linear_output = np.dot(X, self.weights) + self.bias
        prediction = self.activation(linear_output)
        return prediction

    # param X: data to train on
    # param y: actual class data belong to
    def fit(self, X, y):
        for epoch in range(self.epochs):
            for i in range(len(X)):
                prediction = self.predict(X[i])
                error = y[i] - prediction
                # update weights & bias
                self.weights += self.learning_rate * error * X[i]
                self.bias += self.learning_rate * error
            