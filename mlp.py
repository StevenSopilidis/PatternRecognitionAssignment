import numpy as np


# implement neural network as a multilayer perceptron
# we will have the input layer
# one hidden the layer
# and the output
class MLP:
    def __init__(self,input_size,hidden_size,output_size,learning_rate=0.01,epochs=100) -> None:
        self.learning_rate=learning_rate
        self.epochs=epochs

        # wegihts = matrix of size input_size x hidden_size
        self.hidden_weights = np.random.rand(input_size, hidden_size)
        # biases = matrix of size [1 x hidden_size]
        self.hidden_bias = np.zeros(hidden_size)

        self.output_weights = np.random.rand(hidden_size, output_size)
        self.output_bias = np.zeros(output_size)


    # sigmoid activation function
    def activation_hidden(self, x):
        return 1 / (1 + np.exp(-x))
    
    # Using a linear activation function for the output layer (for regression problems)
    def activation_output(self, x):
        return x
    

    # function for predicting value of x
    def predict(self, x):
        hidden_output = self.activation_hidden(np.dot(x, self.hidden_weights) + self.hidden_bias)

        return self.activation_output(np.dot(hidden_output,self.output_weights) + self.output_bias)
    
    def fit(self, X, y):
        for epoch in range(self.epochs):
            for i in range(len(X)):
                # Forward pass
                hidden_output = self.activation_hidden(np.dot(X[i], self.hidden_weights) + self.hidden_bias)
                prediction = self.activation_output(np.dot(hidden_output, self.output_weights) + self.output_bias)

                # Backward pass
                error = y[i] - prediction

                # Update weights and biases for the output layer
                self.output_weights += self.learning_rate * error * hidden_output.reshape(-1, 1)
                self.output_bias += self.learning_rate * error

                # Update weights and biases for the hidden layer
                hidden_error = np.dot(error, self.output_weights.T)
                hidden_delta = hidden_error * hidden_output * (1 - hidden_output)
                self.hidden_weights += self.learning_rate * np.outer(X[i], hidden_delta)
                self.hidden_bias += self.learning_rate * hidden_delta


def RunMLP(updated_folds, house_prices_per_fold, folds):
    network = MLP(9, 30, 1)
    for i in range(0,9):
        network.fit(updated_folds[i], house_prices_per_fold[i])

    sum_mse = 0
    sum_mae = 0
    for i in range(9):
        model_predictions = np.array([network.predict(data) for data in updated_folds[i]])
        actual_prices = np.array(house_prices_per_fold[i])

        sum_mse += np.mean((model_predictions - actual_prices) ** 2)
        sum_mae += np.mean(np.abs(model_predictions - actual_prices))

    print("Training MSE of perceptron algorithm: ", sum_mse/9)
    print("Training MAE of perceptron algorithm: ", sum_mae/9)


    model_predictions = np.array([network.predict(data) for data in updated_folds[9]])
    actual_prices = np.array(folds[9])
    mse = np.mean((model_predictions - actual_prices) ** 2)
    mae = np.mean(np.abs(model_predictions - actual_prices))
    print("Validation MSE of MLP: ", mse)
    print("Validation MAE of MLP: ", mae)