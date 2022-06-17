import pandas as pd
import plotly.express as px
from copy import copy
from scipy import stats
import matplotlib.pyplot as plt
import numpy as np
import plotly.figure_factory as ff
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from tensorflow import keras

# TASK  # 12: TRAIN AN LSTM TIME SERIES MODEL

# Let's test the functions and get individual stock prices and volumes for AAPL
from quotes.MLModels import *

price_volume_df = individual_stock(stock_price_df, stock_vol_df, 'AAPL')

# Get the close and volume data as training data (Input)
training_data = price_volume_df.iloc[:, 1:3].values

# Normalize the data
from sklearn.preprocessing import MinMaxScaler

sc = MinMaxScaler(feature_range=(0, 1))
training_set_scaled = sc.fit_transform(training_data)

# Create the training and testing data, training data contains present day and previous day values
X = []
y = []
for i in range(1, len(price_volume_df)):
    X.append(training_set_scaled[i - 1:i, 0])
    y.append(training_set_scaled[i, 0])

# Convert the data into array format
X = np.asarray(X)
y = np.asarray(y)

# Split the data
split = int(0.7 * len(X))
X_train = X[:split]
y_train = y[:split]
X_test = X[split:]
y_test = y[split:]

# Reshape the 1D arrays to 3D arrays to feed in the model
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
X_train.shape, X_test.shape

# Create the model
inputs = keras.layers.Input(shape=(X_train.shape[1], X_train.shape[2]))
x = keras.layers.LSTM(150, return_sequences=True)(inputs)
x = keras.layers.Dropout(0.3)(x)
x = keras.layers.LSTM(150, return_sequences=True)(x)
x = keras.layers.Dropout(0.3)(x)
x = keras.layers.LSTM(150)(x)
outputs = keras.layers.Dense(1, activation='linear')(x)

model = keras.Model(inputs=inputs, outputs=outputs)
model.compile(optimizer='adam', loss="mse")
model.summary()

# Train the model
history = model.fit(
    X_train, y_train,
    epochs=20,
    batch_size=32,
    validation_split=0.2
)

# Make prediction
predicted = model.predict(X)

# Append the predicted values to the list
test_predicted = []

for i in predicted:
    test_predicted.append(i[0])

df_predicted_lstm = price_volume_df[1:][['Date']]

df_predicted_lstm['predictions'] = test_predicted

# Plot the data
close = []
for i in training_set_scaled:
    close.append(i[0])

df_predicted_lstm['Close'] = close[1:]