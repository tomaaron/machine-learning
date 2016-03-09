import numpy
def J(w,b,x,y):
	z = numpy.dot(x,w)+b
	return (w**2).sum() + numpy.maximum(0,1-y*z).sum()
def DJ(w,b,x,y):
	z = numpy.dot(x,w)+b
	dw = 2*w - (((1-y*z)>0)[:,numpy.newaxis] * x*y[:,numpy.newaxis]).sum(axis=0)
	db = - (((1-y*z)>0)*y).sum(axis=0)
	return dw,db
