
from collections import Counter

def confusion_matrix(data):
	# Implement the function here
	tp=fp=tn=fn=0
	for ele in data:
		y_true, y_pred= ele

		if (y_true==1) & (y_pred==1):
			tp+=1
		if (y_true==0) & (y_pred==0):
			tn+=1
		if (y_true==0) & (y_pred==1):
			fp+=1
		if (y_true==1) & (y_pred==0):
			fn+=1
			
	return [[tp,fn],[fp,tn]]