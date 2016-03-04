import numpy as np
def random_means(d,k):
	return d[np.random.choice(d.shape[0],k)]
def s_eucl_distance(a,b):
	return np.linalg.norm(a-b)**2
def assign_to_means(d,means):
	dist = np.empty([d.shape[0],means.shape[0]])
	for i, m in enumerate(means):
		dist[:,i] = map(lambda x: s_eucl_distance(x,m),d)
	labels =np.argmin(dist,axis=1)
	return dist,labels
def calculate_new_means(labels,d,means):
	for i, m in enumerate(means):
		means[i] = np.mean(d[labels==i],axis=0)
	return means
