from keras import applications
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.utils import np_utils
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Conv2D, MaxPooling2D, Flatten
from keras.layers.normalization import BatchNormalization
from sklearn.model_selection import train_test_split


import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder

reload = False

if reload:
	model = applications.VGG16(include_top=False, weights='imagenet')

	files = os.listdir("images")
	files = [x for x in files if 'jpg' in x]
	files = sorted(files)
	labels = np.array([re.sub("_\d*.jpg", "", x) for x in files])

	le=LabelEncoder()
	labels = le.fit_transform(labels)
	labels = np_utils.to_categorical(labels)

	images = np.array([image.img_to_array(image.load_img(os.path.join('images',file), target_size=(224,224))) for file in files])

	X = preprocess_input(images)

	features = model.predict(X)
	X = features.reshape(features.shape[0], -1)
	y = labels

	np.savez_compressed("features.npz",X=X, y=y)
	print(X.shape, y.shape)
else:
	loaded = np.load("features.npz")
	X = loaded['X']
	y = loaded['y']
	print(X.shape, y.shape)

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2,stratify=y)

	model = Sequential()
	model.add(Dense(256, input_dim=X.shape[1], kernel_initializer='normal'))#, activation='relu'))
	model.add(BatchNormalization())
	model.add(Activation('relu'))
	model.add(Dense(256, kernel_initializer= "normal"))#, activation = 'relu'))
	model.add(BatchNormalization())
	model.add(Activation('relu'))
	model.add(Dense(64, kernel_initializer= "normal"))#, activation = 'relu'))
	model.add(BatchNormalization())
	model.add(Activation('relu'))
	model.add(Dense(32, kernel_initializer= "normal"))#, activation = 'relu'))
	model.add(BatchNormalization())
	model.add(Activation('relu'))
	model.add(Dense(37, kernel_initializer='normal', activation='softmax'))

	model.summary()

	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

	history_model = model.fit(X_train, y_train, batch_size=512, epochs=40, validation_split=.2)

	print(model.evaluate(X_test, y_test))

	df = pd.DataFrame(history_model.history)
	df[['acc', 'val_acc']].plot()
	plt.ylabel("accuracy")
	df[['loss', 'val_loss']].plot(linestyle='--', ax=plt.twinx())
	plt.ylabel("loss")
	plt.title("acc/loss curve")
	plt.savefig("pets-curve.png")

K.clear_session()
