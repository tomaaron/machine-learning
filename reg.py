import numpy as np
import matplotlib.pyplot as plt
def lse(x,y):
	o = np.ones(x.shape[0])
	xo = np.append(x[np.newaxis],o[np.newaxis],axis=0)
	w = np.linalg.inv(xo.dot(xo.T)).dot(xo).dot(y)
	return w
def lse_lambda(x,y,a):
	# linear version
	o = np.ones(x.shape[0])
	xo = np.append(x[np.newaxis],o[np.newaxis],axis=0)
	I = np.identity(2)
	AI=map(lambda x: x*a, I)
	print AI
	w = np.linalg.inv(xo.dot(xo.T)+AI).dot(xo).dot(y)
	return w
def reg(x,y):
	w = lse(x,y)
	line = w[0]*x+w[1]
	plt.plot(x,line)
	plt.scatter(x,y)
	plt.show()
def ridge_reg(x,y,a):
	for i in np.arange(0,a,0.1):
		w = lse_lambda(x,y,i) 	
		line = w[0]*x+w[1]
		plt.plot(x,line,label=str(i))
	plt.scatter(x,y)
	plt.legend()
	plt.show()
