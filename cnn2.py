import pandas as pd
import numpy as np
import sys
import sklearn
import tensorflow as tf
from sklearn.preprocessing import Normalizer
#print(pd.__version__)
#print(np.__version__)
#print(sys.version)
#print(sklearn.__version__)


train_data = pd.read_csv("Training.csv", header=None)
test_data = pd.read_csv("Testing.csv", header=None)

# shape, this gives the dimensions of the dataset
#print('Dimensions of the Training set:', train_data)
#print('Dimensions of the Test set:', test_data)

X = train_data.iloc[:,1:42]
Y = train_data.iloc[:,0]
C = test_data.iloc[:,0]
T = test_data.iloc[:,1:42]

scaler = Normalizer().fit(X)
trainX = scaler.transform(X)

scaler = Normalizer().fit(T)
testT = scaler.transform(T)

y_train = np.array(Y)
y_test = np.array(C)

# reshape input to be [samples, time steps, features]
X_train = np.reshape(trainX, (trainX.shape[0],trainX.shape[1],1))
X_test = np.reshape(testT, (testT.shape[0],testT.shape[1],1))

cnn = tf.keras.Sequential()
cnn.add(tf.keras.layers.Conv1D(64, 3, padding="same",activation="relu",input_shape=(41, 1)))
cnn.add(tf.keras.layers.MaxPooling1D(pool_size=2))
cnn.add(tf.keras.layers.Flatten())
cnn.add(tf.keras.layers.Dense(128, activation="relu"))
cnn.add(tf.keras.layers.Dropout(0.5))
cnn.add(tf.keras.layers.Dense(1, activation="sigmoid"))
print(cnn.summary())

cnn.compile(loss="binary_crossentropy", optimizer="adam",metrics=['accuracy'])

checkpointer = tf.keras.callbacks.ModelCheckpoint(filepath="C:/Users/jazmi/OneDrive/Documents/PFC1/results/checkpoint-{epoch:02d}.hdf5", verbose=1, save_best_only=True, monitor='val_acc',mode='max')
csv_logger = tf.keras.callbacks.CSVLogger('C:/Users/jazmi/OneDrive/Documents/PFC1/results/cnntrainanalysis1.csv',separator=',', append=False)
cnn.fit(X_train, y_train, epochs=100, validation_data=(X_test, y_test),callbacks=[checkpointer,csv_logger])
cnn.save("C:/Users/jazmi/OneDrive/Documents/PFC1/results/cnn_model.hdf5")

loss, accuracy = cnn.evaluate(X_test, y_test)
print("\nLoss: %.2f, Accuracy: %.2f%%" % (loss, accuracy*100))

