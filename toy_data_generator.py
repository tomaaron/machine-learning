import numpy
def gaussian_1d(mean,sigma,n):
	return numpy.random.normal(mean,sigma,n)
def gaussian_2d(mean,sigma,n):
	x,y= numpy.random.multivariate_normal(mean,sigma,n).T
	return x,y
