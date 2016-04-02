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
	return labels
def calculate_new_means(labels,d,means):
	new_means = np.empty_like(means)
	for i, m in enumerate(means):
		if(len(d[labels==i]) > 0):
			new_means[i] = np.mean(d[labels==i],axis=0)
	return new_means
def kmeans(data,k,epsilon=10**-3):
	means_old = random_means(data,k)
	d = sys.maxint 
	while(d > epsilon):
		labels = assign_to_means(data,means_old)
		means_new = calculate_new_means(labels,data,means_old)
		d = s_eucl_distance(means_old,means_new)
		means_old = means_new
	return means_old,labels
