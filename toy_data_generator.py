import numpy as np
import scipy,scipy.io
import matplotlib.pyplot as plt
def rot(a):
	return np.array([[np.cos(a),-np.sin(a)],[np.sin(a),np.cos(a)]]).dot(np.array([[3,0],[0,1]]))
def gaussian_1d(mean,sigma,n):
	return np.random.normal(mean,sigma,n)
def gaussian_2d(mean,sigma,n):
	data = np.empty([n,2])
	x,y= np.random.multivariate_normal(mean,sigma,n).T
	data[:,0]= x
	data[:,1]= y
	return data,x,y 
def toy2d():
	return gaussian_2d([0,0],[[1,0],[0,1]],500)
def cross2d():
	s1 = rot(45)
	s2 = rot(-45)
	d,x,y = toy2d()
	d1 = np.dot(s1,d.T)
	d2 = np.dot(s2,d.T)
	return np.concatenate([d1,d2],axis=1)
def generateData_two_coins(lam, p1, p2, N, M):
	data = np.zeros((N,M),dtype=np.int8)
	probability=0 #
	for i in range(0,N):
	        if(np.random.rand()<=lam):
       			probability=p1 # We are using coin A
		else:
			probability=p2 # We are using coin B              
		data[i] = (np.random.rand(M,1)<=probability)[:,0] #Record the M throws
	return data

def plot(data,distribution):
	N,M = data.shape
        f = plt.figure(figsize=(7,5))
	# Compute max value of the histogram
        sdata = data.sum(axis=1)
	hmax = 1.5*np.max([(sdata == i).sum() for i in range(M+1)])
	# Plot data histogram
	ax1 = f.add_subplot(111)
	ax1.set_ylim(0,hmax)
	ax1.hist(data.sum(axis=1),bins=np.arange(M+2)-0.5,alpha=0.3,color='g')
	ax1.set_xlabel('x')
	ax1.set_xlim(-0.5,M+0.5)
	# Plot probability function
	ax2 = ax1.twinx()
	ax2.set_ylim(0,hmax/N)
	ax2.plot(range(M+1),distribution,'-o',c='r')
	ax2.set_ylabel('p(x)')
	ax2.set_xlim(-0.5,M+0.5)
	plt.show()
