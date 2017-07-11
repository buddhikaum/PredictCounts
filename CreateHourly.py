import json
import numpy as np


#To create sequences of 10 hour

sortedlist = np.loadtxt('sortedindices.txt',dtype=int)

for i in range(0,10):

	Seq_Size=10
	#print str(sortedlist(i))
	fp = open('./alldata/'+str(sortedlist[i])+'.txt','r')
	all_xy = json.load(fp)
	fp.close()
	#print all_xy
	if len(all_xy)<Seq_Size+1:
		continue
	for k in range(0,len(all_xy)-Seq_Size-1):
		#print all_xy[0+k:Seq_Size+k]
		for j in range(0,Seq_Size):
			print all_xy[k+j][0],
		print all_xy[k+Seq_Size][0],'\n',
		for j in range(0,Seq_Size):
			print all_xy[k+j][1],
		print 0,'\n',

