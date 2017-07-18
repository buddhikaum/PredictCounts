import json
import numpy as np
from os.path import expanduser


#To create sequences of 10 hour

sortedlist = np.loadtxt('sortedindices.txt',dtype=int)

def PredictNext():

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

def IsViral():
	
	Seq_Size=5
	for i in range(1000,1100):
		#print str(sortedlist(i))
		#fp = open('/Users/janith/Desktop/alldata/'+str(sortedlist[i])+'.txt','r')
		fp = open(expanduser('~')+'/Desktop/alldata/'+str(sortedlist[i])+'.txt','r')
		all_xy = json.load(fp)
		fp.close()
		#print all_xy
		if len(all_xy)<Seq_Size+1:
			continue
		init_ct =0	
		final_sum =0
		for j in range(0,Seq_Size):
			print all_xy[j][0],
			init_ct += all_xy[j][0]
		for j in range(0,len(all_xy)):
			final_sum += all_xy[j][0]
		if final_sum>1000:
			print '1'
		else:
			print '0'
		for j in range(0,Seq_Size):
			print all_xy[j][1],
		print 0,'\n',

		print 'final sum ',final_sum,'init sum ',init_ct
		#for k in range(0,len(all_xy)-Seq_Size-1):
		#	#print all_xy[0+k:Seq_Size+k]
		#	for j in range(0,Seq_Size):
		#		print all_xy[k+j][0],
		#	print all_xy[k+Seq_Size][0],'\n',
		#	for j in range(0,Seq_Size):
		#		print all_xy[k+j][1],
		#	print 0,'\n',

def ViralTime():


	#for i in rage(0,sortedlist.shape[0]):
	for i in range(0,100):

		fp = open(expanduser('~')+'/Desktop/alldata/'+str(sortedlist[i])+'.txt','r')
		all_xy = json.load(fp)
		fp.close()
		init_ct =0	
		final_sum =0
		#for j in range(0,Seq_Size):
		#	print all_xy[j][0],
		#	init_ct += all_xy[j][0]
		t_min=-1
		is_update = True
		for j in range(0,len(all_xy)):
			final_sum += all_xy[j][0]
			if final_sum>100 and is_update:
				is_update= False
				t_min= j
				#break 
		ngh_t = -1
		if t_min != -1:
			ngh_sum=0
			for j in range(0,len(all_xy)):
				ngh_sum += all_xy[j][1]
				if j==t_min:
					ngh_t = ngh_sum	
			print 'len ',len(all_xy),t_min,ngh_t/ngh_sum

def main():

	#IsViral()
	ViralTime()


if __name__== "__main__":

	main()
