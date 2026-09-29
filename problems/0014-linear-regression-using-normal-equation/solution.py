import numpy as np
def linear_regression_normal_equation(X: list[list[float]], y: list[float]) -> list[float]:
	# Your code here, make sure to round
	X= np.array(X)
	X_T= X.T
	inverse_mat= np.linalg.inv(X_T @ X )
	rest= X_T @ y
	theta= inverse_mat @ rest

	return theta