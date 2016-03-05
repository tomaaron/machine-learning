import numpy as np
def pca(data,k):
	cdata = data -np.mean(data,axis=0)
	cov = np.cov(cdata.T)
	w,v = np.linalg.eigh(cov)
	W = v[:,np.argsort(w)[::-1][:k]]
	H = cdata.dot(W)
	return W,H
	
