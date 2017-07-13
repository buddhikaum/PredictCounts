
import numpy as np
import matplotlib.pyplot as plt
import time
from keras.models import Sequential
from keras.layers.core import Activation, Dense
from keras.layers.recurrent import LSTM
from keras.optimizers import SGD

#def GenTrainData(data_len,n_samples):
Seq_Size = 10
def GenTrainData():

	#dc = .5
	#X = np.zeros((n_samples,1,3*data_len/4))
	#t = np.linspace(0,20*np.pi,data_len);
	#Y = np.zeros((n_samples,data_len/4))
	#for i in range(0,n_samples):
	#	amp = np.random.rand(1)*2
	#	#print amp
	#	full_Vec = dc +np.sin(amp*t)/4
	#	X[i,0,:] = full_Vec[0:data_len*3/4]
	#	Y[i,:] = full_Vec[data_len*3/4:data_len]
	#
	#return X,Y
	train_perc=.5
	alldata = np.genfromtxt('xystructured100.txt')
	#alldata = alldata - alldata.mean()
	n_data = alldata.shape[0]/2
	n_train = int(n_data*train_perc)*2
	print '# Training samples ',n_train/2
	n_test = n_data*2-n_train
	print '# Testing samples ',n_test/2
	Xtemp = alldata[0:n_train,0:Seq_Size]
	Y = alldata[range(0,n_train,2),Seq_Size]
	Xtemp_test = alldata[n_train:n_data*2,0:Seq_Size]
	Ytest = alldata[range(n_train,n_data*2,2),Seq_Size]
	X = np.zeros((n_train/2,Seq_Size,2))
	Xtest = np.zeros((n_test/2,Seq_Size,2))
	#Y = np.zeros((n_train/2,1,1))
	for i in range(0,n_train/2):
		#ct_sum = np.sum(Xtemp[2*i,:])
		X[i,:,0] = Xtemp[2*i,:]
		#ngh_sum = np.sum(Xtemp[2*i+1,:])
		X[i,:,1] = Xtemp[2*i+1,:]
	X[:,:,0] -= X[:,:,0].mean()
	X[:,:,1] -= X[:,:,1].mean()
	for i in range(0,n_test/2):
		ct_sum = np.sum(Xtemp_test[2*i,:])
		Xtest[i,:,0] = Xtemp_test[2*i,:]
		ngh_sum = np.sum(Xtemp_test[2*i+1,:])
		Xtest[i,:,1] = Xtemp_test[2*i+1,:]
	Xtest[:,:,0] -= Xtest[:,:,0].mean()
	Xtest[:,:,1] -= Xtest[:,:,1].mean()

	Y -= Y.mean()
	Ytest -= Ytest.mean()
	#print 'Train input \n',X
	print 'Train input size \n',X.shape
	#print 'Train output \n',Y
	print 'Train output size \n',Y.shape
	#print 'Test input \n',Xtest
	#print 'Test output \n',Ytest
	return X,Y,Xtest,Ytest




def GenTestData(data_len):

	dc = .5
	#amp = np.random.rand(1)/2
	amp = .5
	X = np.zeros((1,1,3*data_len/4))
	t = np.linspace(0,20*np.pi,data_len);
	full_Vec = dc +np.sin(amp*t)/4
	X[0,0:data_len*3/4] = full_Vec[0:data_len*3/4]
	#X[0,0,:] = dc +np.sin(amp*t)/4
	y = full_Vec[data_len*3/4:data_len]

	return y,X

#GenTrainData()
	
#data_len = 1024
#n_samples = 800 # Number of train samples
#
model = Sequential()
model.add(LSTM(1,input_shape=(Seq_Size,2),activation='tanh',return_sequences=False))
##model.add(LSTM(1, input_shape=(1,), activation='tanh',
##					recurrent_activation='hard_sigmoid'))
#model.add(LSTM(Seq_Size))
model.add(Dense(Seq_Size, activation='sigmoid'))
model.add(Dense(Seq_Size, activation='tanh'))
model.add(Dense(1, activation='linear'))
#
#t = np.linspace(0,2*np.pi,data_len);
#amp = .4
#noise_var = .01
#
X,Y,Xtest,Ytest = GenTrainData()
#
sgd = SGD(lr=0.0001, decay=1e-5, momentum=0.9, nesterov=False)
model.compile(loss='mean_squared_error', optimizer=sgd)
#model.compile(loss='mean_absolute_error', optimizer=sgd)
#
#print 'Input \n',X
t1 = time.time()
history = model.fit(X, Y, epochs=1000, batch_size=100, verbose=1)
t2 = time.time()
print model.get_weights()[0]
print model.get_weights()[1]
print model.get_weights()[2]
np.savetxt('loss.txt',history.history['loss'])
#
#y,X_test = GenTestData(data_len)
##X = dc+amp*np.sin(t-np.pi/2)/2
#print 'test input \n',Xtest
Op =  model.predict(Xtest)
XX = np.zeros((1,Seq_Size,2))
#print 'allzeros input ',model.predict(XX)
print 'True out ',Ytest
print 'Predicted out ',Op
print 'Training time ',t2-t1
#output_sin, = plt.plot(t[3*data_len/4:data_len],Op[0,:],label='op')
#input_sin, = plt.plot(t[3*data_len/4:data_len],y,label='true ')
#plt.legend(handles=[output_sin, input_sin])
#plt.show()
