#!/usr/bin/env python3

from datetime import date
import matplotlib.pyplot as plt
import pandas as pd
from_file = __import__('2-from_file').from_file

df = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')

# remove column Weighted_Price
df = df.drop(["Weighted_Price"], axis=1)
# Convert the timestamp values to datatime values
df["Timestamp"] = pd.to_datetime(df['Timestamp'], unit='s')
# data from 2017 and beyond
df = df[df["Timestamp"].dt.year >= 2017]
# set to 0
df["Volume_(BTC)"].fillna(0, inplace=True)
df["Volume_(Currency)"].fillna(0, inplace=True)

# set to the previous row’s
df["High"] = df["Close"].fillna(method = "ffill")
df["Low"] = df["Close"].fillna(method = "ffill")
df["Open"] = df["Close"].fillna(method="ffill")
df["Close"] = df["Close"].fillna(method="ffill")

# Rename the column Timestamp to Date
df.rename(columns={'Timestamp': 'Date'}, inplace=True)
df = df.set_index('Date')

new_df = pd.DataFrame()
new_df['High'] = df.High.resample('D').max()
new_df['Low'] = df.Low.resample('D').min()
new_df['Open'] = df.Open.resample('D').min()
new_df['Close'] = df.Close.resample('D').max()
new_df['Volume_(BTC)'] = df['Volume_(BTC)'].resample('D').sum()
new_df['Volume_(Currency)'] = df['Volume_(Currency)'].resample('D').sum()

new_df.plot()
plt.show()