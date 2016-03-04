import numpy
def gaussian_1d(mean,sigma,n):
	return numpy.random.normal(mean,sigma,n)
def gaussian_2d(mean,sigma,n):
	data = numpy.empty([n,2])
	x,y= numpy.random.multivariate_normal(mean,sigma,n).T
	data[:,0]= x
	data[:,1]= y
	return data
