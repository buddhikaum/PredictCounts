import numpy as np
from MyKeras import GenTrainData
import sys
#import statsmodels.discrete.discrete_model as poi
#import statsmodels.tools.tools as st_tool
import sklearn.linear_model as lr
from sklearn.preprocessing import Binarizer as bn
from keras.models import Sequential
from keras.layers.core import Activation, Dense
from keras.optimizers import SGD
from keras.optimizers import RMSprop

Seq_Size = 10
def Train(X,Y):
	#nodelist = np.genfromtxt("int.txt")
	#T = int(sys.argv[1])
	#for i in nodelist:


	#reg = np.genfromtxt('reg_'+str(int(i))+'.txt')
	poi_model = lr.LinearRegression(fit_intercept=True)
	poi_model.fit(X,Y)
	print 'params ',poi_model.get_params()
	print 'coeff ',poi_model.coef_
	print 'intercept ',poi_model.intercept_
	reg = np.c_[X, np.ones(X.shape[0])]
	#meas = np.genfromtxt('meas_15_'+str(int(i))+'.txt')
	#print 'lable ',i
	#print 'before reg',reg
	#reg = st_tool.add_constant(reg,prepend=False)	
	#print 'after reg',reg
	#raw_input()
	#P1 = poi.Poisson(Y[1:100],reg[1:100,:])
	#lim = 1
	#bd = ((-lim,lim),(-lim,lim),(-lim,lim),(-lim,lim))
	#F = P1.fit(method='lbfgs',bounds=bd)
	#F = P1.fit(method='bfgs',disp=True)
	#print F.summary()
	#print F.params
	#print F.mle_retvals
	#np.savetxt('regx1_'+str(int(i))+'.txt',F.params)
	#P2 = poi.Poisson(meas[1:T,1],reg)
	#F = P2.fit(method='lbfgs',bounds=bd)
	#np.savetxt('regx2_'+str(int(i))+'.txt',F.params)
	#P3 = poi.Poisson(meas[1:T,2],reg)
	#F = P3.fit(method='lbfgs',bounds=bd)
	#np.savetxt('regx3_'+str(int(i))+'.txt',F.params)
	return poi_model

def DLmethod(X,Y,Xtest,Ytest):


	model = Sequential()
	#model.add(Dense(Seq_Size,input_shape=(Seq_Size,), activation='linear'))
	model.add(Dense(1,input_shape=(Seq_Size,),activation='sigmoid',use_bias=True,
					bias_initializer='zeros',kernel_initializer='zeros'))
	#model.add(Dense(Seq_Size, activation='linear'))
	#model.add(Dense(1, activation='linear',use_bias=True))
	rms = RMSprop(lr=0.01,decay=1e-5)
	sgd = SGD(lr=1e-3, decay=1e-8, momentum=0,nesterov=False)
	Xperm = np.random.permutation(X.shape[0])
	X = X[Xperm,:]
	Y = Y[Xperm]
	model.compile(loss='binary_crossentropy', optimizer=rms)
	history = model.fit(X, Y, epochs=500, batch_size=3000, verbose=1)
	np.savetxt('loss.txt',history.history['loss'])
	print 'all weights ',model.get_weights()
	#print 'weights ',model.get_weights()[0]
	yhat = model.predict(Xtest)
	ybinary = np.zeros(yhat.shape,dtype=int)
	for k in range(0,yhat.shape[0]):
		if yhat[k,0]>=.5:
			ybinary[k,0] =1	
		else :
			ybinary[k,0] = 0
	np.savetxt('yhat.txt',yhat)
	np.savetxt('ybinary.txt',ybinary,fmt='%d')
	print 'er ',np.sum(np.abs(ybinary[:,0]-Ytest))

def main():

	X_temp,Y,Xtest_temp,Ytest,ymean = GenTrainData()
	
	X = np.zeros((X_temp.shape[0],Seq_Size*2))
	for i in range(0,X_temp.shape[0]):
		X[i,0:Seq_Size] = X_temp[i,:,0]
		X[i,Seq_Size:2*Seq_Size] = X_temp[i,:,1]
		#print X[i,:]
	X_test = np.zeros((Xtest_temp.shape[0],Seq_Size*2))
	for i in range(0,Xtest_temp.shape[0]):
		X_test[i,0:Seq_Size] = Xtest_temp[i,:,0]
		X_test[i,Seq_Size:2*Seq_Size] = Xtest_temp[i,:,1]
	##print Y+ymean
	
	X = np.zeros((X_temp.shape[0],Seq_Size))
	for i in range(0,X_temp.shape[0]):
		X[i,0:Seq_Size] = X_temp[i,:,0]
		#X[i,Seq_Size:2*Seq_Size] = X_temp[i,:,1]
		#print X[i,:]
	X_test = np.zeros((Xtest_temp.shape[0],Seq_Size))
	for i in range(0,Xtest_temp.shape[0]):
		X_test[i,0:Seq_Size] = Xtest_temp[i,:,0]
		#X_test[i,Seq_Size:2*Seq_Size] = X_temp[i,:,1]

	#poi_model = Train(X,Y)
	#yhat = poi_model.predict(X_test)
	#print poi_model.coef_
	#print 'Regression yhat shape ',yhat.shape
	#for i in range(0,yhat.shape[0]):
	#	print np.dot(poi_model.coef_,X_test[i,:]),Ytest[i]
	#	print np.multiply(poi_model.coef_,X_test[i,:])
	#	print poi_model.coef_,X_test[i,:]
	#np.savetxt('yhat.txt',yhat)
	#print np.sum(np.abs(yhat-Ytest))

	DLmethod(X,Y,X_test,Ytest)

if __name__== "__main__":

	main()
