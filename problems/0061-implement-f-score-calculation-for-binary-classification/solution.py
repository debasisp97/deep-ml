import numpy as np

def f_score(y_true, y_pred, beta):
	"""
	Calculate F-Score for a binary classification task.

	:param y_true: Numpy array of true labels
	:param y_pred: Numpy array of predicted labels
	:param beta: The weight of precision in the harmonic mean
	:return: F-Score rounded to three decimal places
	"""
	tp= ((y_true==1)&(y_pred==1)).sum()
	fp= ((y_true==0)&(y_pred==1)).sum()
	fn= ((y_true==1)&(y_pred==0)).sum()
	
	p= tp/(tp+fp)
	r= tp/(tp+fn)
	
	nmr= (p*r)
	const= (1+beta**2)
	dnr= ((beta**2)* p)+r
	
	f= const * nmr/dnr
	return round(f,4)
