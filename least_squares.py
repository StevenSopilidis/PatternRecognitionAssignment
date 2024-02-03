import numpy as np

# X: training data
# Y: expected output
def LeastSquares(X, Y):
    x = np.array(X)
    y = np.array(Y)

    # Add a column of ones for the intercept term
    x_with_intercept = np.column_stack((np.ones(len(x)), x))

    x_t  = x_with_intercept.transpose()
    x_t_x = np.dot(x_t, x_with_intercept)
    x_t_x_inv = np.linalg.inv(x_t_x)
    x_t_y = np.dot(x_t, y)
    coff = np.dot(x_t_x_inv, x_t_y)

    return coff

def RunLeastSquares(folds, updated_folds, house_prices_per_fold):
    # make predictions using Least Squares algorithm
    coefficients = []
    for i in range(9):
        prices_per_fold = []
        coefficients.append(LeastSquares(updated_folds[i], house_prices_per_fold[i]))

    avg_coefficients = np.mean(coefficients, axis=0)

    new_data = np.array(updated_folds[9])
    new_data_with_intercept = np.column_stack((np.ones(len(new_data)), new_data))

    sum_mse = 0
    sum_mae = 0
    for i in range(9):
        data = np.array(updated_folds[i])
        data_with_intercept = np.column_stack((np.ones(len(data)), data))
        model_predictions = np.dot(data_with_intercept, avg_coefficients)
        actual_prices = np.array(house_prices_per_fold[i])

        sum_mse += np.mean((model_predictions - actual_prices) ** 2)
        sum_mae += np.mean(np.abs(model_predictions - actual_prices))

    print("Training MSE of Least Squares: ", sum_mse/10)
    print("Training MAE of Least Squares: ", sum_mae/10)


    predictions = np.dot(new_data_with_intercept, avg_coefficients)
    actual_prices = [row[8] for row in folds[9]]
    mse_least_squares = np.mean((predictions - actual_prices) ** 2)
    mae_least_squares = np.mean(np.abs(predictions - actual_prices))
    print("MSE of least squares algorithm: ", mse_least_squares)
    print("MAE of least squares algorithm: ", mae_least_squares)