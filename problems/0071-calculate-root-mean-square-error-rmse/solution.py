
import numpy as np

def rmse(y_true, y_pred):
	# Write your code here

	rmse_res=None
	m= y_true.shape[0]
	n= y_pred.shape[0]

	if not m or not n:
		raise Exception("Can not be empty")

	if m!= n:
		raise Exception("Mismatch size")
	
	diff= y_true - y_pred
	# sq_diff= diff**2
	# ssr= np.mean(sq_diff)
	# rmse_res= np.sqrt(ssr)

	rmse_res = np.sqrt( np.mean(diff**2))
	return round(rmse_res,3)
