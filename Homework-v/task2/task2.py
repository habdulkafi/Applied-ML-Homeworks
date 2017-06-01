
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from keras import backend as K
from keras.utils import np_utils
import matplotlib.pyplot as plt

import pandas as pd
from sklearn.datasets import fetch_mldata
from sklearn.model_selection import train_test_split, GridSearchCV, ShuffleSplit


mnist = fetch_mldata('MNIST original')
X = mnist.data
y = np_utils.to_categorical(mnist.target)

# X_train, X_test, y_train, y_test = train_test_split(mnist.data, y, test_size=.2)

# print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)


def baseline_model(hidden_size = 256):
	# create model
	model = Sequential()
	model.add(Dense(hidden_size, input_dim=784, kernel_initializer='normal', activation='relu'))
	model.add(Dense(int(hidden_size/2), kernel_initializer= "normal", activation = 'relu'))
	model.add(Dense(10, kernel_initializer='normal', activation='softmax'))
	# Compile model
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	print(model.summary())
	return model



def dropout_model(hidden_size = 1024, dropout_rate=.5):
	# create model
	model = Sequential()
	model.add(Dense(hidden_size, input_dim=784, kernel_initializer='normal', activation='relu'))
	model.add(Dropout(dropout_rate))
	model.add(Dense(int(hidden_size/2), kernel_initializer= "normal", activation = 'relu'))
	model.add(Dropout(dropout_rate))
	model.add(Dense(10, kernel_initializer='normal', activation='softmax'))
	# Compile model
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	print(model.summary())
	return model



# history_callback = estimator.fit(X, y, validation_split=.2)


vanilla = False

if vanilla:
	params = {"hidden_size": [32, 128, 256, 512]}
	estimator = KerasClassifier(build_fn=baseline_model, epochs=50, batch_size=512)
	grid = GridSearchCV(estimator, param_grid=params, n_jobs=1, 
		cv=ShuffleSplit(test_size=0.20, n_splits=1, random_state=0) )

	grid_result = grid.fit(X, y)

	print(grid.best_params_)

	print(grid.cv_results_)

	df = pd.DataFrame(grid.cv_results_)
	df.to_pickle("vanilla.pkl")
	plt.plot(df.param_hidden_size, df.mean_test_score)
	plt.title("Hidden Size vs Test Score")
	plt.ylabel("accuracy")
	plt.xlabel("Hidden Size")
	plt.savefig("vanilla.png")

else:
	params = {"hidden_size": [32, 128, 256, 512],
			  "dropout_rate": [.25, .5]}
	estimator = KerasClassifier(build_fn=dropout_model, epochs=50, batch_size=512)
	grid = GridSearchCV(estimator, param_grid=params, n_jobs=1, 
		cv=ShuffleSplit(test_size=0.20, n_splits=1, random_state=0) )

	grid_result = grid.fit(X, y)

	print(grid.best_params_)

	print(grid.cv_results_)

	df = pd.DataFrame(grid.cv_results_)
	df.to_pickle("dropout.pkl")
	do25 = df[df["param_dropout_rate"] == .25]
	do50 = df[df["param_dropout_rate"] == .50]
	plt.plot(do50.param_hidden_size, do50.mean_test_score)
	plt.plot(do25.param_hidden_size, do25.mean_test_score)
	plt.legend(["dropout .50", "dropout .25"])
	plt.title("Hidden Size vs Test Score")
	plt.ylabel("accuracy")
	plt.xlabel("Hidden Size")
	plt.savefig("drop_out.png")

	# df = pd.DataFrame(history_callback.history)


	# df[['acc', 'val_acc']].plot()
	# plt.ylabel("accuracy")
	# df[['loss', 'val_loss']].plot(linestyle='--', ax=plt.twinx())
	# plt.ylabel("loss")


K.clear_session()

