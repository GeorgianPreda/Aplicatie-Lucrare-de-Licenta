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
from quotes.MLModels import *
from sklearn.linear_model import Ridge

# Note that Ridge regression performs linear least squares with L2 regularization.
# Create and train the Ridge Linear Regression  Model

regression_model = Ridge()
regression_model.fit(X_train, y_train)

# Test the model and calculate its accuracy
lr_accuracy = regression_model.score(X_test, y_test)
print("Linear Regression Score: ", lr_accuracy)

# Make Prediction
predicted_prices = regression_model.predict(X)

# Append the predicted values into a list
Predicted = []
for i in predicted_prices:
    Predicted.append(i[0])

len(Predicted)

# Append the close values to the list
close = []
for i in price_volume_target_scaled_df:
    close.append(i[0])

# Create a dataframe based on the dates in the individual stock data
df_predicted_ridge = price_volume_target_df[['Date']]

# Add the close values to the dataframe
df_predicted_ridge['Close'] = close

# Add the predicted values to the dataframe
df_predicted_ridge['Prediction'] = Predicted