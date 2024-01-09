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