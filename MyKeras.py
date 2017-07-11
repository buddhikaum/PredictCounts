
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers.core import Activation, Dense
from keras.layers.recurrent import LSTM
from keras.optimizers import SGD

def GenTrainData(data_len,n_samples):

	dc = .5
	X = np.zeros((n_samples,1,3*data_len/4))
	t = np.linspace(0,20*np.pi,data_len);
	Y = np.zeros((n_samples,data_len/4))
	for i in range(0,n_samples):
		amp = np.random.rand(1)*2
		#print amp
		full_Vec = dc +np.sin(amp*t)/4
		X[i,0,:] = full_Vec[0:data_len*3/4]
		Y[i,:] = full_Vec[data_len*3/4:data_len]
	
	return X,Y

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

	
data_len = 1024
n_samples = 800 # Number of train samples

model = Sequential()
model.add(LSTM(1,input_shape=(1,data_len*3/4)))
#model.add(LSTM(1, input_shape=(1,), activation='tanh',
#					recurrent_activation='hard_sigmoid'))
model.add(Dense(data_len/4, activation='sigmoid'))

t = np.linspace(0,2*np.pi,data_len);
amp = .4
noise_var = .01

X_train,Y_train = GenTrainData(data_len,n_samples)

sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='mean_squared_error', optimizer=sgd)

history = model.fit(X_train, Y_train, epochs=1000, batch_size=4, verbose=0)

y,X_test = GenTestData(data_len)
#X = dc+amp*np.sin(t-np.pi/2)/2
Op =  model.predict(X_test)
print 'True out ',y
print 'Predicted out ',Op
output_sin, = plt.plot(t[3*data_len/4:data_len],Op[0,:],label='op')
input_sin, = plt.plot(t[3*data_len/4:data_len],y,label='true ')
plt.legend(handles=[output_sin, input_sin])
plt.show()
