import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from copy import copy
from scipy import stats
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go

# Read the stock data csv file, here's the list of the stocks considered:
from seaborn import cm

stocks_df = pd.read_csv(
    'C:/Users/Georgian/Desktop/djangostock/stock_dr.csv')
stocks_df

# Sort the stock data by date
stocks_df = stocks_df.sort_values(by=['Date'])
stocks_df

# Print out the number of stocks
print('Total Number of stocks : {}'.format(len(stocks_df.columns[1:])))

# Print the name of stocks
print('Stocks under consideration are:')

for i in stocks_df.columns[1:]:
    print(i)


# TASK  # 6: CALCULATE MULTIPLE STOCKS DAILY RETURNS


# Let's define a function to calculate stocks daily returns (for all stocks)
def daily_return(df):
    df_daily_return = df.copy()

    # Loop through each stock (while ignoring time columns with index 0)
    for i in df.columns[1:]:

        # Loop through each row belonging to the stock
        for j in range(1, len(df)):
            # Calculate the percentage of change from the previous day
            df_daily_return[i][j] = ((df[i][j] - df[i][j - 1]) / df[i][j - 1]) * 100

        # set the value of first row to zero since the previous value is not available
        df_daily_return[i][0] = 0

    return df_daily_return

# Get the daily returns
stocks_daily_return = daily_return(stocks_df)

# TASK  # 7: CALCULATE THE CORRELATIONS BETWEEN DAILY RETURNS

# Daily Return Correlation

cm = stocks_daily_return.drop(columns=['Date']).corr()

plt.figure(figsize=(10, 10))
ax = plt.subplot()
sns.heatmap(cm, annot=True, ax=ax)
plt.show()
