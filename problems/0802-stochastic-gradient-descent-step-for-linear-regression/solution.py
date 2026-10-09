import numpy as np

def sgd_update(X: np.ndarray, y: np.ndarray, weights: np.ndarray, learning_rate: float, n_iter: int) -> list:
    """
    Perform n_iter steps of stochastic gradient descent on a linear regression
    model with MSE loss, cycling through samples in order.

    Returns the final weight vector as a Python list.
    """
    m,n= X.shape
    y= y.reshape(-1,1)
    w= weights[:]

    for i in range(n_iter):
        idx= i % m
        pred= np.dot(X[idx].T,w) # X[i] :(1,n), W:(n,1)
        error= (pred - y[idx])
        grad= error *  X[idx]
        w -= (learning_rate*2)*grad

    return w.flatten().tolist()
