import numpy as np

def accuracy_score(y_true, y_pred):
	# Your code here
	accuracy_score=0
	accuracy_score= (y_true==y_pred).sum()/y_true.shape[0]
	return accuracy_score