#!/usr/bin/env python3
'''forecast btc '''
import tensorflow as tf
import tensorflow.keras as K
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

preprocess = __import__('preprocess_data').pre_process

X_train, y_train, x_test, y_test, sc = preprocess()

model = K.models.Sequential()
# Adding the first LSTM layer and some Dropout regularisation
model.add(K.layers.LSTM(units=50, return_sequences=True,
                        input_shape=(X_train.shape[1], 1)))
model.add(K.layers.Dropout(0.2))
# Adding a second LSTM layer and some Dropout regularisation
model.add(K.layers.LSTM(units=50, return_sequences=True))
model.add(K.layers.Dropout(0.2))
# Adding a third LSTM layer and some Dropout regularisation
model.add(K.layers.LSTM(units=50, return_sequences=True))
model.add(K.layers.Dropout(0.2))
# Adding a fourth LSTM layer and some Dropout regularisation
model.add(K.layers.LSTM(units=50))
model.add(K.layers.Dropout(0.2))
# Adding the output layer
model.add(K.layers.Dense(units=1))
# Compiling the RNN
model.compile(optimizer='adam', loss='mean_squared_error')

# use a tf.data.Dataset to feed data to your model
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
train_dataset = train_dataset.shuffle(100).batch(32)
test_dataset = test_dataset.batch(32)

# Fitting the RNN to the Training set
model.fit(train_dataset, epochs=10)
# evaluate
model.evaluate(test_dataset)

# make predictions
predicted_btc_price = model.predict(test_dataset)
predicted_btc = sc.inverse_transform(predicted_btc_price)
y = sc.inverse_transform(y_test)

# plot
plt.plot(y, color='red', label='Real BTC Price')
plt.plot(predicted_btc, color='blue', label='Predicted BTC Price')
plt.title('BTC Price Prediction')
plt.xlabel('Time')
plt.ylabel('BTC Price')
plt.legend()
plt.show()
