import numpy as np
def precision(y_true, y_pred):
	# Your code here
	p=0
	tp= ((y_true==1) & (y_pred==1)).sum()
	fp= ((y_true==0) & (y_pred==1)).sum()
	dnr=tp+fp
	p= tp/dnr if dnr else 0
	return p
