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



def RunPerceptron(folds, house_prices_per_fold, medians, min_value, max_value, min_values, max_values):
    # for the perceptron we will cluster the data into two categories
    # 1. Above the median price (1) 
    # 2. Below the median price (0)
    median_price_scaled = ((medians[8] - min_values[8]) / (max_values[8] - min_values[8])) * (max_value - min_value) + min_value


    # convert the data in -1 and 1 manner
    training_data_per_fold = []
    for row in house_prices_per_fold:
        # get the price which will be the training data
        training_data_per_fold.append([-1 if data < median_price_scaled else 1 for data in row])


    # for the first 9 folds train the perceptron algorithm
    model = Perceptron(9, 0.01, 160)
    for i in range(9):
        model.fit(np.array(folds[i]), training_data_per_fold[i])

    # calculate avg MSE AND MAE FOR TRAINED DATA
    sum_mse = 0
    sum_mae = 0
    for i in range(9):
        model_predictions = np.array([model.predict(data) for data in folds[i]])
        actual_prices = np.array(training_data_per_fold[i])

        sum_mse += np.mean((model_predictions - actual_prices) ** 2)
        sum_mae += np.mean(np.abs(model_predictions - actual_prices))

    print("Training MSE of perceptron algorithm: ", sum_mse/9)
    print("Training MAE of perceptron algorithm: ", sum_mae/9)

    # calculate the MSE (MEAN SQUARED ERROR)
    # and the MAE (MEAN ABSOLUTE ERROR)
    model_predictions = np.array([model.predict(data) for data in folds[9]])
    mse = np.mean((model_predictions - actual_prices) ** 2)
    mae = np.mean(np.abs(model_predictions - actual_prices))
    print("Validation MSE of perceptron algorithm: ", mse)
    print("Validation MAE of perceptron algorithm: ", mae)