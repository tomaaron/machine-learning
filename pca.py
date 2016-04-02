import numpy as np
import toy_data_generator as tdg
import matplotlib.pyplot as plt
def pca(data,k):
	''' computes k principle components
	Input:	data - NxD array of N data points with dimension D
		k - k number of principle components to be computed
	Output: W -  
		H - 
	'''
	cdata = data -np.mean(data,axis=0)
	cov = np.cov(cdata.T)
	w,v = np.linalg.eigh(cov)
	W = v[:,np.argsort(w)[::-1][:k]]
	H = cdata.dot(W)
	return W,H
def toy_pca():
	data,x,y = tdg.toy2d()
	s = tdg.rot(-45)
	data=s.dot(data.T).T
	w,h=pca(data,2)
	ax = plt.axes()
	ax.arrow(0, 0, w[0][0], w[1][0], head_width=0.05, head_length=0.1, fc='r', ec='r')
	ax.arrow(0, 0, w[0][1], w[1][1], head_width=0.05, head_length=0.1, fc='r', ec='r')
	ax.scatter(data[:,0],data[:,1])
	plt.show()
