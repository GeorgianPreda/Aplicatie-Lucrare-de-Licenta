import pandas as pd
import seaborn as sns
import plotly.express as px
from copy import copy
from scipy import stats
import matplotlib.pyplot as plt
import numpy as np
import plotly.figure_factory as ff
import plotly.graph_objects as go

# Read the stock data file
stocks_df = pd.read_csv(
    'C:/Users/Georgian/Desktop/djangostock/stock.csv')
stocks_df

# Sort the data based on Date
stocks_df = stocks_df.sort_values(by=['Date'])
stocks_df


# Function to normalize the prices based on the initial price
def normalize(df):
    x = df.copy()
    for i in x.columns[1:]:
        x[i] = x[i] / x[i][0]
    return x


##########TASK  # 3: CALCULATE DAILY RETURNS


# Function to calculate the daily returns
def daily_return(df):
    df_daily_return = df.copy()

    # Loop through each stock
    for i in df.columns[1:]:

        # Loop through each row belonging to the stock
        for j in range(1, len(df)):
            # Calculate the percentage of change from the previous day
            df_daily_return[i][j] = ((df[i][j] - df[i][j - 1]) / df[i][j - 1]) * 100

        # set the value of first row to zero, as previous value is not available
        df_daily_return[i][0] = 0
    return df_daily_return


# Get the daily returns
stocks_daily_return = daily_return(stocks_df)
stocks_daily_return

################TASK  # 4: CALCULATE BETA FOR A SINGLE STOCK

# Select any stock, let's say Apple
stocks_daily_return['AAPL']

# Select the S&P500 (Market)
stocks_daily_return['sp500']

# plot a scatter plot between the selected stock and the S&P500 (Market)

# stocks_daily_return.plot(kind='scatter', x='sp500', y='AAPL')

# Fit a polynomial between the selected stock and the S&P500 (Poly with order = 1 is a straight line)

# beta represents the slope of the line regression line (market return vs. stock return).
# Beta is a measure of the volatility or systematic risk of a security or portfolio compared to the entire market (S&P500)
# Beta is used in the CAPM and describes the relationship between systematic risk and expected return for assets

# Beta = 1.0, this indicates that its price activity is strongly correlated with the market.
# Beta < 1, indicates that the security is theoretically less volatile than the market (Ex: Utility stocks). If the stock is included, this will make the portfolio less risky compared to the same portfolio without the stock.
# Beta > 1, indicates that the security's price is more volatile than the market. For instance, Tesla stock beta is 1.26 indicating that it's 26% more volatile than the market.
# Tech stocks generally have higher betas than S&P500 but they also have excess returns
# MGM is 65% more volatile than the S&P500!


beta, alpha = np.polyfit(stocks_daily_return['sp500'], stocks_daily_return['AAPL'], 1)
print('Beta for {} stock is = {} and alpha is = {}'.format('AAPL', beta, alpha))

# Now let's plot the scatter plot and the straight line on one plot
#stocks_daily_return.plot(kind='scatter', x='sp500', y='AAPL')

# Straight line equation with alpha and beta parameters
# Straight line equation is y = beta * rm + alpha

# plt.plot(stocks_daily_return['sp500'], beta * stocks_daily_return['sp500'] + alpha, '-', color='r')

################ TASK  # 5: APPLY THE CAPM FORMULA TO AN INDIVIDUAL STOCK

beta

# Let's calculate the average daily rate of return for S&P500
stocks_daily_return['sp500'].mean()

# Let's calculate the annualized rate of return for S&P500
# Note that out of 365 days/year, stock exchanges are closed for 104 days during weekend days (Saturday and Sunday)
# Check your answers with: https://dqydj.com/sp-500-return-calculator/
rm = stocks_daily_return['sp500'].mean() * 252
rm

# Assume risk free rate is zero
# Also you can use the yield of a 10-years U.S. Government bond as a risk free rate
rf = 0

# Calculate return for any security (APPL) using CAPM
ER_AAPL = rf + (beta * (rm - rf))

ER_AAPL

#################### TASK  # 6: CALCULATE BETA FOR ALL STOCKS

# Let's create a placeholder for all betas and alphas (empty dictionaries)
beta = {}
alpha = {}

# Loop on every stock daily return
for i in stocks_daily_return.columns:

    # Ignoring the date and S&P500 Columns
    if i != 'Date' and i != 'sp500':
        # plot a scatter plot between each individual stock and the S&P500 (Market)
        #stocks_daily_return.plot(kind='scatter', x='sp500', y=i)

        # Fit a polynomial between each stock and the S&P500 (Poly with order = 1 is a straight line)
        b, a = np.polyfit(stocks_daily_return['sp500'], stocks_daily_return[i], 1)
#####################################
        # plt.plot(stocks_daily_return['sp500'], b * stocks_daily_return['sp500'] + a, '-', color='r')

        beta[i] = b

        alpha[i] = a

        # plt.show()

# Let's view Beta for every stock
beta

# Let's view alpha for each of the stocks
# Alpha describes the strategy's ability to beat the market (S&P500)
# Alpha indicates the “excess return” or “abnormal rate of return,”
# A positive 0.175 alpha for Tesla means that the portfolio’s return exceeded the benchmark S&P500 index by 17%.

alpha


#################### TASK  # 7: APPLY CAPM FORMULA TO CALCULATE THE RETURN FOR THE PORTFOLIO

# # Obtain a list of all stock names
# keys = list(beta.keys())
# keys
#
# # Define the expected return dictionary
# ER = {}
#
# rf = 0  # assume risk free rate is zero in this case
# rm = stocks_daily_return['sp500'].mean() * 252  # this is the expected return of the market
# rm
#
# for i in keys:
#     # Calculate return for every security using CAPM
#     ER[i] = rf + (beta[i] * (rm - rf))
#
# for i in keys:
#     print('Expected Return Based on CAPM for {} is {}%'.format(i, ER[i]))
#
# # Assume equal weights in the portfolio
# portfolio_weights = 1 / 8 * np.ones(8)
# portfolio_weights
#
# # Calculate the portfolio return
# ER_portfolio = sum(list(ER.values()) * portfolio_weights)
# ER_portfolio
#
# print('Expected Return Based on CAPM for the portfolio is {}%\n'.format(ER_portfolio))

def printcapm():
    # Plot the data
    # interactive_plot(df_predicted, "Original Vs Prediction - LSTM Model")
    # Obtain a list of all stock names
    keys = list(beta.keys())
    keys

    # Define the expected return dictionary
    ER = {}

    rf = 0  # assume risk free rate is zero in this case
    rm = stocks_daily_return['sp500'].mean() * 252  # this is the expected return of the market
    rm
    output_capm = []
    for i in keys:
        # Calculate return for every security using CAPM
        ER[i] = rf + (beta[i] * (rm - rf))

    for i in keys:
        output_capm.append('Expected Return Based on CAPM for {} is {}%'.format(i, ER[i]))

    # Assume equal weights in the portfolio
    portfolio_weights = 1 / 8 * np.ones(8)
    portfolio_weights

    # Calculate the portfolio return
    ER_portfolio = sum(list(ER.values()) * portfolio_weights)
    ER_portfolio
    output_capm.append('Expected Return Based on CAPM for the portfolio is {}%\n'.format(ER_portfolio))
    return output_capm
