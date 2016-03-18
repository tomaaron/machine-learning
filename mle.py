import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
def gauss_mle_1d(data):
	mu = np.mean(data,axis=0)
	sigma = np.sqrt(np.mean(map(lambda x: (x-mu)**2,data)))
	return mu,sigma
#http://stackoverflow.com/questions/14873203/plotting-of-1-dimensional-gaussian-distribution-function
def gaussian(x, mu, sig):
	    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

def naive_bayes(w1train,w2train,test):
	# prior
	n = w1train.shape[0]+w2train.shape[0]
	w_1 = w1train.shape[0] / float(n)
	w_2 = w2train.shape[0] / float(n)
	print 'prior w1:', w_1
	print 'prior w2:', w_2
	# likelihood
	mu_1, s_1 = gauss_mle_1d(w1train)
	mu_2, s_2 = gauss_mle_1d(w2train)
	post_1 = gaussian(test,mu_1,s_1)
	post_2 = gaussian(test,mu_2,s_2)
	print 'p(w1|x)=',post_1
	print 'p(w2|x)=',post_2
	p_1 = post_1*w_1
	p_2 = post_2*w_2
	print 'class 1',p_1
	print 'class 2',p_2
	print 'bla_1', p_1 / (p_1+p_2)
	print 'bla_2', p_2 / (p_1+p_2)
	x1 = np.linspace(-3,8,100)
	plt.title('Epic Info')
	plt.ylabel('Y axis')
	plt.xlabel('X axis')
	plt.plot(x1,mlab.normpdf(x1,mu_1,s_1),label='estimate class1')
	plt.plot(x1,mlab.normpdf(x1,mu_2,s_2),label='estimate class2')
	plt.legend()
	plt.text(-2,0.7,'class 1:%s\nclass2: %s'%(p_1,p_2))
	plt.plot(test,0,'o',label='test point')
	plt.show()

