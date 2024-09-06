
from tkinter import X
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.someSubpackage import someClassifier
from sklearn import svm
import os
import requests
import timeit
from numba import jit
import django





# X_train, X_test, y_train, y_test = train_test_split(X, y,
# test_size = 0.5)




#  Partitioning the Data

# np.random.seed(1234)
# X=np.pi*(2*np.random.random(size=(400,2))-1)
# y=(np.cos(X[:,0])*np.sin(X[:,1])>=0)
# X_train , X_test , y_train , y_test = train_test_split(X, y,
# test_size=0.5)
# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.scatter(X_train[y_train==0,0],X_train[y_train==0,1], c='g',
# marker='o',alpha=0.5)
# ax.scatter(X_train[y_train==1,0],X_train[y_train==1,1], c='b',
# marker='o',alpha=0.5)
# ax.scatter(X_test[y_test==0,0],X_test[y_test==0,1], c='g',
# marker='s',alpha=0.5)
# ax.scatter(X_test[y_test==1,0],X_test[y_test==1,1], c='b',
# marker='s',alpha=0.5)
# plt.savefig('sklearntraintest.pdf',format='pdf')
# plt.show()


   #  Standardization
# min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
# x_scaled = min_max_scaler.fit_transform(X)
# # equivalent to:
# x_scaled = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))



# Fitting and Prediction
# clf = someClassifier() # choose appropriate classifier
# clf.fit(X_train, y_train) # fit the data
# y_prediction = clf.predict(X_test) # predict



# testing the model
# clf = svm.SVC(kernel = 'rbf')
# clf.fit(X_train , y_train)
# y_prediction = clf.predict(X_test)
# from sklearn.metrics import confusion_matrix
# print(confusion_matrix(y_test , y_prediction))



#  System Calls, URL Access, and Speed-Up

# for c in "123456":
# try: # if it does not yet exist
# os.mkdir("MyDir"+ c) # make a directory
# except: # otherwise
# pass # do nothing
# uname = "https://github.com/DSML-book/Programs/tree/master/
# Appendices/Python Primer/"
# fname = "ataleof2cities.txt"
# r = requests.get(uname + fname)
# print(r.text)
# open('MyDir1/ato2c.txt', 'wb').write(r.content) #write to a file
# bytes mode is important here




# n = 10**8
# #@jit
# def myfun(s,n):
#   for i in range(1,n):
#     s = s+ 1/i
#     return s
# start = timeit.time.clock()
# print("Euler's constant is approximately {:9.8f}".format(
# myfun(0,n) - np.log(n)))
# end = timeit.time.clock()
# print("elapsed time: {:3.2f} seconds".format(end-start))

