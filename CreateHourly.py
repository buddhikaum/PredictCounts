import json
import numpy as np
from os.path import expanduser
import matplotlib.pyplot as plt


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
	
	Seq_Size=10
	#print 'shape 0',sortedlist.shape[0]
	P = np.random.permutation(sortedlist.shape[0])
	#for i in range(0,sortedlist.shape[0]):
	for i in P:
	#for i in range(0,10):
		#print str(sortedlist(i))
		#fp = open('/Users/janith/Desktop/alldata/'+str(sortedlist[i])+'.txt','r')
		fp = open(expanduser('~')+'/Desktop/minfiles/'+'min_'+str(sortedlist[i])+'.txt','r')
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
		if final_sum>100:
			print '1'
		else:
			print '0'
		for j in range(0,Seq_Size):
			print all_xy[j][1],
		print -1,'\n',

		#print 'final sum ',final_sum,'init sum ',init_ct
		#for k in range(0,len(all_xy)-Seq_Size-1):
		#	#print all_xy[0+k:Seq_Size+k]
		#	for j in range(0,Seq_Size):
		#		print all_xy[k+j][0],
		#	print all_xy[k+Seq_Size][0],'\n',
		#	for j in range(0,Seq_Size):
		#		print all_xy[k+j][1],
		#	print 0,'\n',

def ViralTime():


	for i in range(0,sortedlist.shape[0]):
	#for i in range(0,1000):

		fp = open(expanduser('~')+'/Desktop/minfiles/'+'min_'+str(sortedlist[i])+'.txt','r')
		all_xy = json.load(fp)
		fp.close()
		init_ct =0	
		final_sum =0.0
		#for j in range(0,Seq_Size):
		#	print all_xy[j][0],
		#	init_ct += all_xy[j][0]
		t_min=-1
		is_update = True
		cnt_10=0.0
		for j in range(0,len(all_xy)):
			final_sum += all_xy[j][0]
		for j in range(0,len(all_xy)):
			#final_sum += all_xy[j][0]
			init_ct += all_xy[j][0]
			if init_ct>=.5*final_sum:
			#if init_ct>=100:
				is_update= False
				t_min= j
				break 
			#if j==9:
			#	t_min = j
			#	cnt_10 = final_sum
		ngh_t = -1
		if t_min != -1:
			ngh_sum=0
			for j in range(0,len(all_xy)):
				ngh_sum += all_xy[j][1]
				if j==t_min:
					ngh_t = ngh_sum	
			print len(all_xy),t_min,cnt_10/final_sum,ngh_t/ngh_sum

def PlotTweetCt():

	plt.figure(1)
	F = np.genfromtxt('fact_min.txt')
	bins = np.linspace(0,1000,101)
	print bins 
	H = np.histogram(F[:,1],bins)
	#n,bins,patches = plt.hist(F[:,1],bins)
	plt.bar(bins[0:100]/10,H[0],width=1,linewidth=2)
	print H[1].shape
	print H[0].shape
	print H[1][1:101].shape
	print H[0]
	print np.sum(H[0])
	#plt.plot(H[1][1:101],H[0],'r--')
	#plt.xlim((0,10))
	plt.grid(True)
	#plt.xlabel('Hours from initial Tweet')
	plt.xlabel('Time taken to reach half of the total Retweets (hours)')
	plt.ylabel('Number of Tweets')
	#plt.title('Number of Different Tweets against time taken to reach half of the to')
	#plt.draw()
	plt.show(block=False)

def PlotNeighCt():

	plt.figure(2)
	F = np.genfromtxt('fact_min.txt')
	bins = np.linspace(0,1,11)
	print bins 
	H = np.histogram(F[:,3],bins)
	#n,bins,patches = plt.hist(F[:,1],bins)
	plt.bar(bins[0:10],H[0],width=.1,linewidth=2)
	print H[1].shape
	print H[0].shape
	print H[1][1:101].shape
	print H[0]
	print np.sum(H[0])
	#plt.plot(H[1][1:101],H[0],'r--')
	#plt.xlim((0,10))
	plt.grid(True)
	#plt.xlabel('Hours from initial Tweet')
	plt.xlabel('Ratio between sum of the followers at half way and sum of followers at the end')
	plt.ylabel('Number of Tweets')
	plt.title('Number of Different Tweets against the fraction of user that saw the tweet at half way')
	#plt.draw()
	plt.show(block=False)
def main():

	#IsViral()
	#ViralTime()
	PlotTweetCt()
	PlotNeighCt()
	print ' Enter any key to exit '
	raw_input()


if __name__== "__main__":

	main()
