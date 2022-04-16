# # define input sequence being log returns
# raw_seq = train['Close']
#
# # choose a number of time steps
# n_steps = 5
#
# # split into samples
# X, y = split_sequence(raw_seq, n_steps)
#
# scaler = MinMaxScaler()
# scaled_train = scaler.fit_transform(X)
# scaler_y = MinMaxScaler()
# scaled_y = scaler_y.fit_transform(y.reshape(-1,1))
# test_y = y
#
# X,y = scaled_train, scaled_y
#
# # Push the items into
# 	# [
# 	# 	[
# 	# 		[1],[2],[3],[4],[5]
# 	# 	],
# 	# ]
#
# n_features = 1
# X = X.reshape((X.shape[0], X.shape[1], n_features))
#
#




#
# print('RUNNING MODEL PREDICTION')
# # pred = model.predict(X)
#
#
# #y will contain each value of n[0]+5 shifted
# #pred will contain ~ what y should be in each segment
#
# # ------------------------------------------------
# # define input sequence being log returns
# raw_seq = test['Close']
#
# # choose a number of time steps
# n_steps = 5
#
# # split into samples
# X, y = split_sequence(raw_seq, n_steps)
#
# scaler = MinMaxScaler()
# scaled_test = scaler.fit_transform(X)
# scaler_y = MinMaxScaler()
# scaled_y = scaler_y.fit_transform(y.reshape(-1,1))
#
# X,y = scaled_test, scaled_y
#
# # Push the items into
# 	# [
# 	# 	[
# 	# 		[1],[2],[3],[4],[5]
# 	# 	],
# 	# ]
#
# n_features = 1
# X = X.reshape((X.shape[0], X.shape[1], n_features))
# # ---------------------------------------------
#
# print(test)
# exit()
# pred = model.predict(X)
# inverse_pred = scaler_y.inverse_transform(pred);
# inverse_y = scaler_y.inverse_transform(y);
#
# #y = x increments
#
# date = test['Datetime'][6:]
#
# fig, ax = plt.subplots()
# ax.plot_date(date, inverse_y, '-b')
# ax.plot_date(date, inverse_pred, '-r')
# plt.show()
