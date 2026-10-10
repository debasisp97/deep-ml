import numpy as np

def train_logreg(X: np.ndarray, y: np.ndarray, learning_rate: float, iterations: int) -> tuple[list[float], ...]:
	"""
	Gradient-descent training algorithm for logistic regression, optimizing parameters with Binary Cross Entropy loss.
	"""
	# Your code here
	m,n= X.shape
	X_new= np.column_stack([np.ones(m), X])
	w= np.zeros((n+1,))
	loss=[]

	for i in range(iterations) :
		z = X_new @ w
		p = 1/(1+ np.exp(-z))
		error = p-y
		loss.append(-np.sum( y * np.log(p) + (1-y)*np.log(1-p) ))
		
		grad= (X_new.T @ error)
		w= w- learning_rate * grad 
	
	return w.round(4), np.array(loss).round(4)
