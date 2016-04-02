import numpy as np
import toy_data_generator as tdg
import sys
from scipy.stats import binom
import scipy
def distribution(data,lam,p1,p2):
	n,m = data.shape
	result = np.zeros(m+1);
	for i in range(len(result)):
		result[i] = lam * binom.pmf(i,m,p1) + (1-lam) * binom.pmf(i,m,p2)
	return result
def loglikelihood(data,lam,p1,p2):
	n,m = data.shape
	return sum([np.log(lam*p1**(sum(i))*(1-p1)**(m-sum(i))+(1-lam)*p2**(sum(i))*(1-p2)**(m-sum(i))) for i in data])/float(n)
def EM(data,lam=.5,p1=.25,p2=.75):
	n,m = data.shape
	ll = loglikelihood(data,lam,p1,p2)
	diff = 1
	it = 0
	print 'before: %.3f'%(ll)
	while ( diff > 0.001 ):
		lam_new = sum([r1(lam,p1,p2,sum(i),m-sum(i)) for i in data])/float(n)
		p1_new = sum([r1(lam,p1,p2,sum(i),m-sum(i))*sum(i) for i in data])/(m*sum([r1(lam,p1,p2,sum(i),m-sum(i)) for i in data]))
		p2_new = sum([r2(lam,p1,p2,sum(i),m-sum(i))*sum(i) for i in data])/(m*sum([r2(lam,p1,p2,sum(i),m-sum(i)) for i in data]))
		ll_new = loglikelihood(data,lam_new,p1_new,p2_new)
		diff = ll_new - ll
		it += 1
		ll  = ll_new
		lam = lam_new
		p1  = p1_new
		p2  = p2_new
		print 'it:%2d  lambda: %.2f  p1: %.2f  p2: %.2f  log-likelihood: %.3f'%(it, lam, p1, p2, ll)
def unknownData():
		return scipy.io.loadmat('data.mat')['data']
def r1(lam,p1,p2,hxi,txi):
	return lam*p1**hxi*(1-p1)**(txi)/(lam*p1**hxi*(1-p1)**txi+(1-lam)*p2**hxi*(1-p2)**(txi))
def r2(lam,p1,p2,hxi,txi):
	return (1-lam)*p2**hxi*(1-p2)**(txi)/(lam*p1**hxi*(1-p1)**txi+(1-lam)*p2**hxi*(1-p2)**(txi))
