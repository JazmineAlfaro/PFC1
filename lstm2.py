from __future__ import print_function
import os
import pandas as pd
import numpy as np
np.random.seed(1337)  # for reproducibility
from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Embedding
from keras.layers import LSTM, SimpleRNN, GRU
from keras.datasets import imdb
from keras.utils.np_utils import to_categorical
from sklearn.metrics import (precision_score, recall_score,f1_score, accuracy_score,mean_squared_error,mean_absolute_error)
from sklearn import metrics
from sklearn.preprocessing import Normalizer
import h5py
from keras import callbacks
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger

traindata = pd.read_csv('Training.csv', header=None)
testdata = pd.read_csv('Testing.csv', header=None)


X = traindata.iloc[:,1:42]
Y = traindata.iloc[:,0]
C = testdata.iloc[:,0]
T = testdata.iloc[:,1:42]

trainX = np.array(X)
testT = np.array(T)

trainX.astype(float)
testT.astype(float)

scaler = Normalizer().fit(trainX)
trainX = scaler.transform(trainX)

scaler = Normalizer().fit(testT)
testT = scaler.transform(testT)

y_train = np.array(Y)
y_test = np.array(C)


X_train = np.array(trainX)
X_test = np.array(testT)


batch_size = 64

# 1. define the network
model = Sequential()
model.add(Dense(1024,input_dim=41,activation='relu'))
model.add(Dropout(0.01))
model.add(Dense(1))
model.add(Activation('sigmoid'))

print('getcwd:      ', os.getcwd())
path = os.getcwd()

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

checkpointer = callbacks.ModelCheckpoint(filepath=path+"/kddresults/checkpoint-{epoch:02d}.hdf5", verbose=1, save_best_only=True, monitor='loss')
csv_logger = CSVLogger(path + '/kddresults/training_set_dnnanalysis.csv',separator=',', append=False)
model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=2, callbacks=[checkpointer,csv_logger])
model.save(path + "/kddresults/dnn1layer_model.hdf5")

loss, accuracy = model.evaluate(X_test, y_test)
print("\nLoss: %.2f, Accuracy: %.2f%%" % (loss, accuracy*100))
#Loss: 0.53, Accuracy: 92.19%