import numpy as np 
import sys
import json


SEC_PER_DAY=360
#SEC_PER_DAY=100 

def main():

	row_indices = np.loadtxt('index.csv',delimiter=',')
	all_data = np.loadtxt('data.csv',delimiter=',')
	i=0		
	for kk in range(0,row_indices.shape[0]):
		end_row = row_indices[kk,3]
		n_row = all_data.shape[0]
		#if end_row>n_row:
		#	break
		#print 'n_row ',end_row
		all_xy = []
		j=1
		while i<end_row :
			crnt_tw_ct =0
			crnt_ngh_ct =0
			while i<end_row and all_data[i,0]<=j*SEC_PER_DAY :
				crnt_tw_ct +=1
				crnt_ngh_ct += all_data[i,1]
				i+=1
				#print 'i ',i
			#print 'end time ',all_data[i,0]
			#i +=1
			j +=1
			all_xy.append([crnt_tw_ct,crnt_ngh_ct])
		fp = open(str(kk)+'.txt','w')
		json.dump(all_xy,fp)
		fp.close()
	#print all_xy

if __name__== "__main__":

	main()
