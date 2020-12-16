#!/usr/bin/env python3
'''preprocess data'''
import os
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler


def pre_process():
    '''
    zip_path = tf.keras.utils.get_file(
        origin='https://drive.google.com/u/0/open?id=16MgiuBfQKzXPoWFWi2w-LKJuZ7LgivpE',
        fname='coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv.zip',
        extract=True)
    '''
    bitstamp = pd.read_csv('bitstampUSD_1-min_data_2012-01-01_to_2020-04-22.csv')
    '''bitstamp["Timestamp"] = pd.to_datetime(bitstamp['Timestamp'], unit='s')
    bitstamp["Open"] = bitstamp["Open"].fillna(method = "ffill")
    bitstamp["High"] = bitstamp["High"].fillna(method = "ffill")
    bitstamp["Low"] = bitstamp["Low"].fillna(method = "ffill")'''
    bitstamp["Close"] = bitstamp["Close"].fillna(method = "ffill")
    bitstamp.drop(["Timestamp", "Open", "High", "Low", "Volume_(BTC)", "Volume_(Currency)"], axis=1)
    # bitstamp.head()
    # Feature Scaling
    sc = MinMaxScaler(feature_range = (0, 1))
    training_set = sc.fit_transform(bitstamp)
    del bitstamp

    result = []
    for index in range(len(training_set) - 60):
        result.append(training_set[index: index + 60])

    result = np.array(result)
    training_set = result[:training_set.shape[0] // 2]
    test_set = result[training_set.shape[0] // 2:]

    # Creating a data structure with 60 time-steps and 1 output
    X_train = training_set[:, :-1]
    y_train = training_set[:, -1]
    x_test = test_set[:, :-1]
    y_test = test_set[:, -1]
    # X_train, y_train = np.array(X_train), np.array(y_train)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    return X_train, y_train, x_test, y_test
