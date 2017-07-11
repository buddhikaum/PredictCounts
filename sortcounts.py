import numpy as np 


all_cts = np.genfromtxt('./alldata/allcounts.txt',dtype=int)
G = all_cts
#print 'G ',G
#np.savetxt('sortedindices.txt',G[G[:,1].argsort()]
Indices = G[G[:,1].argsort()[::-1]]

for i in range(0,Indices.shape[0]):
	print str(Indices[i,0])
