
import numpy as np

def r_squared(y_true, y_pred):
	# Write your code here

	ssr= np.sum((y_true-y_pred)**2)
	sst= np.sum((y_true-np.mean(y_true))**2)
	r2= 1-(ssr/sst)
	return  r2
