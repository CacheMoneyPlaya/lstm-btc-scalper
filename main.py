# Get historical BTC data

# Push into dataframe

# Grab closes and normalize data, potentially standardize instead

# Break data into two groupings, groups of 5 sequences of close price + expected 6th

# Create LSTM model

# Run training data against 85% of the dataframe

# Run the model against the remaining 15% and plot all data vs predicted last 15%

import pandas as pd
import numpy as np
from numpy import array
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
#from scipy.stats import kurtosis,skew
#from scipy import stats
import yfinance as yf

#from keras.models import Sequential
#from keras.layers import LSTM
#from keras.layers import Dense

#Get the data and push into dataframe, resetting index as a standard list
pair = 'BTC-USD'
data = yf.download(tickers=pair, period="7d", interval="1m")
data['log_return'] = np.log1p(data['Close'].pct_change())
data = data.dropna(subset=['log_return'])
data.reset_index(inplace=True)
train, test = train_test_split(data, test_size=0.2, shuffle=False)
sns.lineplot(x=train['Datetime'], y=train['Close'], data=train, palette=['green'])
sns.lineplot(x=test['Datetime'], y=test['Close'], data=test, palette=['red'])

# split a univariate sequence into samples
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


# define input sequence
raw_seq = train['log_return']

# choose a number of time steps
n_steps = 5
# split into samples
X, y = split_sequence(raw_seq, n_steps)

scaler = MinMaxScaler()
scaled_train = scaler.fit_transform(X)
scaler_y = MinMaxScaler()
scaled_y = scaler_y.fit_transform(y.reshape(-1,1))


print(scaled_train)

