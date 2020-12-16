#!/usr/bin/env python3
''' '''

preprocess= __import__('preprocess_data').pre_process

X_train, y_train, x_test, y_test = preprocess()
print(X_train)
print(x_test)