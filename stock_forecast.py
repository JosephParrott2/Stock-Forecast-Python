# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 13:11:47 2020

@author: wicke
"""

from __future__ import print_function

import datetime
import numpy as np
import pandas as pd
import sklearn
import yfinance as yf

from sklearn.ensemble import RandomForestClassifier 
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import confusion_matrix
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.svm import LinearSVC, SVC

def create_lagged_series(symbol, start_date, end_date, lags = 5):
    
    # Obtain stock information from Yahoo Finance
    ts = yf.download(symbol, start = start_date-datetime.timedelta(days=365), end = end_date)
    
    # Create the lagged DataFrame
    tslag = pd.DataFrame(index=ts.index)
    tslag["Today"] = ts["Adj Close"]
    tslag["Volume"] = ts["Volume"]
    
    # Create the shifted lag series of prior trading period close values
    for i in range(0, lags):
        tslag["Lag%s" % str(i+1)] = ts["Adj Close"].shift(i+1)
        
    # Create the returns DataFrame
    tsret = pd.DataFrame(index=tslag.index)
    tsret["Volume"] = tslag["Volume"]
    tsret["Today"] = tslag["Today"].pct_change()*100.0
    
    # If any values of %returns equal zero, set them to a small number
    for i, x in enumerate(tsret["Today"]):
        if (abs(x) < 0.0001):
            tsret["Today"][i] = 0.0001
            
    # Create the lagged %returns column
    for i in range(0, lags):
        tsret["Lag%s" % str(i+1)] = \
        tslag["Lag%s" % str(i+1)].pct_change()*100.0
        
    # Create the Direction column (+1 or -1) for up/down day
    tsret["Direction"] = np.sign(tsret["Today"])
    tsret = tsret[tsret.index >= start_date]
    
    return tsret

if __name__ == "__main__":
    
    #Create a lagged series of the S&P500 US stock market index
    snpret = create_lagged_series("^GSPC", datetime.datetime(2001,1,10), datetime.datetime(2005,12,31), lags=5)
    
    # Use the prior 2 days of returns as predictor
    X = snpret[["Lag1", "Lag2"]]
    y = snpret["Direction"]
    
    # Split data into two parts: before and after 01/01/2005
    start_test = datetime.datetime(2005,1,1)
    
    # Create training and test sets
    X_train = X[X.index < start_test]
    X_test = X[X.index >= start_test]
    y_train = y[y.index < start_test]
    y_test = y[y.index >= start_test]

    # Creae the parameterised models
    print("Hit Rates/Confusion Matrices:\n")
    models = [("LR", LogisticRegression()), ("LDA", LinearDiscriminantAnalysis()), ("QDA", QuadraticDiscriminantAnalysis()),("LSVC",LinearSVC()), ("RSVM", SVC(C=1000000.0, cache_size=200, class_weight=None,coef0=0.0, degree=3, gamma=0.0001, kernel="rbf",max_iter=-1,probability=False,random_state=None,shrinking=True,tol=0.001,verbose=False)), ("RF",RandomForestClassifier(n_estimators=1000,criterion="gini",max_depth=None, min_samples_split=2,min_samples_leaf=1, max_features="auto",bootstrap=True,oob_score=False,n_jobs=1,random_state=None,verbose=0))]

    # Iterate through the models
    for m in models:
        
        # Train each of the models
        m[1].fit(X_train,y_train)
        
        # Make an array of predictions on the test set
        pred = m[1].predict(X_test)
        
        # Output the hit-rate and the confusion matrix
        print("%s:\n%0.3f" % (m[0], m[1].score(X_test, y_test)))
        print("%s\n" % confusion_matrix(pred,y_test))
    
    