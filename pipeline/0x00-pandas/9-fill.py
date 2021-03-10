#!/usr/bin/env python3

import pandas as pd
from_file = __import__('2-from_file').from_file

df = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')
# remove column Weighted_Price
df = df.drop(["Weighted_Price"], axis=1)
# set to 0
df["Volume_(BTC)"].fillna(0, inplace=True)
df["Volume_(Currency)"].fillna(0, inplace=True)

# set to the previous row’s
df["High"] = df["Close"].fillna(method = "ffill")
df["Low"] = df["Close"].fillna(method = "ffill")
df["Open"] = df["Close"].fillna(method="ffill")
df["Close"] = df["Close"].fillna(method="ffill")


print(df.head())
print(df.tail())