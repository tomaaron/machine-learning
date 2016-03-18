import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
def train_krr(X_train,Y_train,kwidth,llambda):
	K = gaussian_kernel(X_train,X_train,kwidth)
	I = np.identity(X_train.shape[1])
	a = np.linalg.inv(K+I*llambda).dot(Y_train.T)
	return a
def apply_krr(a,X_test,X_train,kwidth):
	Kt = gaussian_kernel(X_test,X_train,kwidth)
	Y_est = Kt.dot(a).T
	return Y_est
def gaussian_kernel(X,Y,kwidth):
	K=cdist(X.T,Y.T,'euclidean')
	K = np.exp(K ** 2 / -2.*kwidth**2)
	return K
def sin_test_krr():
	X_train = sp.arange(0,10,.01)
	X_train = X_train[None,:]
	Y_train = sp.sin(X_train) + sp.random.normal(0,.5,X_train.shape)
	kws = [0.1,1,10]
	lls = [10**(-10),1,500]
	colors = ['r', 'b', 'g']
	plt.subplot(2, 1, 1)
	plt.plot(X_train.T,Y_train.T,'+k')
	plt.title('KRR:testing various kernel widths with fixed lambda')
	for k,i in zip(kws,colors):
		a = train_krr(X_train,Y_train,k,1)
		Y_est = apply_krr(a,X_train,X_train,k)
		plt.plot(X_train.T, Y_est.T,c=i, linewidth = 2, label = 'kwidth='+str(k))
	plt.legend()	
	plt.subplot(2, 1, 2)	
	plt.plot(X_train.T,Y_train.T,'+k')
	plt.title('KRR:testing various lambdas with fixed kernel width')
	for l,i in zip(lls,colors):
		a = train_krr(X_train,Y_train,1,l)
		Y_est = apply_krr(a,X_train,X_train,1)
		plt.plot(X_train.T, Y_est.T,c=i, linewidth = 2, label = 'lambda='+str(l))
	plt.legend()
	plt.show()
