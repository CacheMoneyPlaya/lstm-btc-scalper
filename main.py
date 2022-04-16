import numpy as np
import pandas as pd
import yfinance as yf
import seaborn as sns
from numpy import array
import tensorflow as tf
from os.path import exists
from keras.layers import LSTM
from keras.layers import Dense
import matplotlib.pyplot as plt
from keras.models import load_model
from keras.models import Sequential
from keras.models import model_from_json
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

test = None
train = None
model = None

def build_data_sets():
	global test, train
	pair = 'BTC-USD'
	data = yf.download(tickers=pair, period="7d", interval="1m")
	data['log_return'] = np.log1p(data['Close'].pct_change())
	data = data.dropna(subset=['Close'])
	data.reset_index(inplace=True)
	train, test = train_test_split(data, test_size=0.2, shuffle=False)



def split_sequence(sequence, n_steps):
	X, y = list(), list()
	for i in range(len(sequence)):
		# find the end of this pattern
		end_ix = i + n_steps
		# check if we are beyond the sequence
		if end_ix >= len(sequence)-1:
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequence.iloc[i:end_ix], sequence.iloc[end_ix]
		X.append(seq_x)
		y.append(seq_y)
	return array(X), array(y)

def build_model():
	global model
	if not exists('model_save'):
		model = Sequential()
		model.add(LSTM(100, activation='swish', input_shape=(n_steps, n_features), return_sequences=True))
		model.add(LSTM(50, activation = 'swish'))
		model.add(Dense(1))
		model.compile(optimizer='adam', loss='mse', metrics = 'mse')
		model.fit(X,y,epochs=100)
		model_json = model.to_json()

		with open("model_num.json", "w") as json_file:
			json_file.write(model_json)

		model.save("model_save")
	else:
		model = load_model('model_save')

tf.config.run_functions_eagerly(True)
build_data_sets()
build_model()


fiveHTest = test['Close'][:500]
tests = np.array(fiveHTest).reshape(-1,1)
scaler = MinMaxScaler()
scaledFirst500 = scaler.fit_transform(tests)
date = test['Datetime'][:500]

track = []
first5 = scaledFirst500[:5]

for i, x in enumerate(scaledFirst500):
	# Reshape
	X = first5.reshape((1, 5, 1))
	# predict
	pred = model.predict(X)
	# push prediction into payload
	first5 = np.append(first5, pred[0])
	# push removed value to track i.e. calcd values
	track.append(first5[0])
	# delete first from paylaod
	first5 = np.delete(first5, 0)

ax.plot_date(date, scaledFirst500, '-b')
ax.plot_date(date, track, '-r')
plt.show()
