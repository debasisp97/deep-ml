import numpy as np

def mini_batch_gd_step(X: np.ndarray, y: np.ndarray, weights: np.ndarray, bias: float, batch_indices: list, lr: float) -> np.ndarray:
    """
    Perform one mini-batch gradient descent update step for linear regression with MSE loss.
    Returns a 1D array of length D+1: updated weights followed by updated bias.
    """
    m,n= X.shape
    sz= len(batch_indices)

    X_new= X[batch_indices]
    y_new= y[batch_indices]

    pred= np.dot(X_new, weights) + bias
    error = (pred-y_new)

    grad_weight= (2/sz)*  np.dot(X_new.T, error )
    grad_bais= (2/sz) * np.sum(error)

    weights= weights - lr * grad_weight
    bias= bias - lr * grad_bais

    # print(weights, bias)
    return np.concatenate((weights, np.array([bias])), axis=None)