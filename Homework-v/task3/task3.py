import scipy.io as sio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Conv2D, MaxPooling2D, Flatten
from keras.layers.normalization import BatchNormalization
from keras import backend as K




train = sio.loadmat("train_32x32.mat")
test = sio.loadmat("test_32x32.mat")

X_train = np.rollaxis(train['X'], 3, 0)/255
y_train = np_utils.to_categorical(train['y'])

X_test = np.rollaxis(test['X'], 3, 0)/255
y_test = np_utils.to_categorical(test['y'])

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

input_shape = (32, 32, 3)

cnn = Sequential()

cnn.add(Conv2D(64, kernel_size = (3,3),
			   # activation='relu',
			   input_shape=input_shape))
cnn.add(BatchNormalization())
cnn.add(Activation('relu'))
cnn.add(MaxPooling2D(pool_size=(2,2)))
cnn.add(Conv2D(32, (3,3)))#, activation='relu'))
cnn.add(BatchNormalization())
cnn.add(Activation('relu'))
cnn.add(MaxPooling2D(pool_size=(2,2)))
cnn.add(Conv2D(32, (3,3)))#, activation='relu'))
cnn.add(BatchNormalization())
cnn.add(Activation('relu'))
cnn.add(MaxPooling2D(pool_size=(2,2)))
cnn.add(Flatten())
cnn.add(Dense(128))#, activation='relu'))
cnn.add(BatchNormalization())
cnn.add(Activation('relu'))
cnn.add(Dropout(.25))
cnn.add(Dense(128))#, activation='relu'))
cnn.add(BatchNormalization())
cnn.add(Activation('relu'))
cnn.add(Dropout(.50))
cnn.add(Dense(128))#, activation='relu'))
cnn.add(BatchNormalization())
cnn.add(Activation('relu'))
cnn.add(Dropout(.25))
cnn.add(Dense(64))#, activation='relu'))
cnn.add(BatchNormalization())
cnn.add(Activation('relu'))
cnn.add(Dropout(.50))
cnn.add(Dense(11, activation='softmax'))

cnn.summary()

cnn.compile("adam", "categorical_crossentropy", metrics=['accuracy'])


history_cnn = cnn.fit(X_train, y_train, batch_size=512, epochs=40, validation_split=.1)

score = cnn.evaluate(X_test, y_test)

print(score)

df = pd.DataFrame(history_cnn.history)
df[['acc', 'val_acc']].plot()
plt.ylabel("accuracy")
df[['loss', 'val_loss']].plot(linestyle='--', ax=plt.twinx())
plt.ylabel("loss")
plt.title("acc loss curve for conv 64 + max pool 2x2 + \n conv 32 + max pool 2x2 + conv 32 + max pool 2x2 + \n dense 128 + dense 128 + dense 128 + dense 64 \nwith batch norm + drop out")
plt.savefig("loss_acc_curve.png")

K.clear_session()
