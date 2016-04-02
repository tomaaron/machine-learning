import numpy as np
from numpy import linalg as LA 
def whiten(D):
	# center data
	D_centered = D - np.mean(D,axis=1)[:,np.newaxis]
	# whitening data
	C = np.cov(D_centered)
	U,s,Vt = np.linalg.svd(C)
	D = np.diag(1. / np.sqrt(s))
	W = np.dot(np.dot(U,D),Vt.T)
	D_whiten= np.dot(W,D_centered) 
	return D_whiten,W 
def fastICA(x,alpha):
	x_whiten,w = whiten(x)
	wt=np.ones(2)
	w_o = np.identity(2)
	for i in range(1500):
		wtx = np.dot(wt,x_whiten)
		gwtx = g(wtx,alpha)
		wt = np.mean(np.array(gwtx*x_whiten),axis=1) - np.mean(g1(wtx,alpha))*wt
		wt = wt / LA.norm(wt)
		dif = w_o - wt
		w_o = wt
		print LA.norm(dif)
	return wt
def m_fastICA(X,a,c):
	X_whiten,W = whiten(X)
	w = np.ones(X.shape[0])
	for p in range(c):
		for i in range(1500):
			wtx = np.dot(wt,X_whiten)
			gwtx = g(wtx,a)
			w[p] = 1/float(X.shape[1])*np.dot(X_whiten,gwtx) - 1/float(X.shape[1])*g1(wtx)
def g1(x,a):
	return a*(1-np.tanh(a*x)**2)
def g(x,a):
	return np.tanh(a*x)
