from keras.models import Sequential
from keras.layers import Dense, Activation
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
from keras.utils import np_utils
from keras.wrappers.scikit_learn import KerasClassifier
from keras import backend as K

import pandas as pd
import matplotlib.pyplot as plt

data = load_iris()
y = np_utils.to_categorical(data.target)

X_train, X_test, y_train, y_test = train_test_split(data.data, y, test_size=.1)

# define baseline model
def baseline_model(hidden_size = 4):
	# create model
	model = Sequential()
	model.add(Dense(hidden_size, input_dim=4, kernel_initializer='normal', activation='relu'))
	model.add(Dense(int(hidden_size/2), kernel_initializer='normal', activation='relu'))
	model.add(Dense(3, kernel_initializer='normal', activation='sigmoid'))
	print(model.summary())
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model


estimator = KerasClassifier(build_fn=baseline_model, epochs=100, batch_size=5)

params = {"hidden_size": [4, 6, 8, 10]}

grid = GridSearchCV(estimator, param_grid=params, n_jobs=1, cv=3)

grid_result = grid.fit(X_train, y_train)

print(grid.best_params_)

# estimator.fit(X_train, y_train)

print(grid.score(X_test, y_test))
df = pd.DataFrame(grid.cv_results_)
df.to_pickle("results.pkl")
print(df)
plt.plot(df["param_hidden_size"], df["mean_test_score"])
plt.ylabel("Average Test Score")
plt.xlabel("Hidden Size")
plt.savefig("fig1.png")


K.clear_session()

