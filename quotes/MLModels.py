import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import plotly.express as px
from copy import copy
from scipy import stats
import matplotlib.pyplot as plt
import numpy as np
import plotly.figure_factory as ff
import plotly.graph_objects as go

# Read stock prices data
stock_price_df = pd.read_csv(
    'C:/Users/Georgian/Desktop/djangostock/stock.csv')
stock_price_df

# Read the stocks volume data
stock_vol_df = pd.read_csv(
    "C:/Users/Georgian/Desktop/djangostock/stock_volume.csv")
stock_vol_df

# Sort the data based on Date
stock_price_df = stock_price_df.sort_values(by=['Date'])
stock_price_df

# Sort the data based on Date
stock_vol_df = stock_vol_df.sort_values(by=['Date'])
stock_vol_df

# Check if Null values exist in stock prices data
stock_price_df.isnull().sum()

# Check if Null values exist in stocks volume data
stock_vol_df.isnull().sum()

# Get stock prices dataframe info
stock_price_df.info()

# Get stock volume dataframe info
stock_vol_df.info()

stock_vol_df.describe()


# TASK  # 3: PERFORM EXPLORATORY DATA ANALYSIS AND VISUALIZATION

# Function to normalize stock prices based on their initial price
def normalize(df):
    x = df.copy()
    for i in x.columns[1:]:
        x[i] = x[i] / x[i][0]
    return x


# Function to plot interactive plots using Plotly Express
def interactive_plot(df, title):
    fig = px.line(title=title)
    for i in df.columns[1:]:
        fig.add_scatter(x=df['Date'], y=df[i], name=i)
    fig.show()

    # plot interactive chart for stocks data

    # interactive_plot(stock_price_df, 'Stock Prices')


# Function to concatenate the date, stock price, and volume in one dataframe
def individual_stock(price_df, vol_df, name):
    return pd.DataFrame({'Date': price_df['Date'], 'Close': price_df[name], 'Volume': vol_df[name]})


# Function to return the input/output (target) data for AI/ML Model
# Note that our goal is to predict the future stock price
# Target stock price today will be tomorrow's price
def trading_window(data):
    # 1 day window
    n = 1

    # Create a column containing the prices for the next 1 days
    data['Target'] = data[['Close']].shift(-n)

    # return the new dataset
    return data


# Let's test the functions and get individual stock prices and volumes for AAPL
price_volume_df = individual_stock(stock_price_df, stock_vol_df, 'AAPL')
price_volume_df

price_volume_target_df = trading_window(price_volume_df)
price_volume_target_df

# Remove the last row as it will be a null value
price_volume_target_df = price_volume_target_df[:-1]
price_volume_target_df

# Scale the data
from sklearn.preprocessing import MinMaxScaler

sc = MinMaxScaler(feature_range=(0, 1))
price_volume_target_scaled_df = sc.fit_transform(price_volume_target_df.drop(columns=['Date']))

price_volume_target_scaled_df

price_volume_target_scaled_df.shape

# Creating Feature and Target
X = price_volume_target_scaled_df[:, :2]
y = price_volume_target_scaled_df[:, 2:]

# Converting dataframe to arrays
# X = np.asarray(X)
# y = np.asarray(y)
X.shape, y.shape

# Spliting the data this way, since order is important in time-series
# Note that we did not use train test split with it's default settings since it shuffles the data
split = int(0.65 * len(X))
X_train = X[:split]
y_train = y[:split]
X_test = X[split:]
y_test = y[split:]

X_train.shape, y_train.shape

X_test.shape, y_test.shape


# Define a data plotting function
def show_plot(data, title):
    plt.figure(figsize=(13, 5))
    plt.plot(data, linewidth=3)
    plt.title(title)
    plt.grid()


# show_plot(X_train, 'Training Data')
# show_plot(X_test, 'Testing Data')




