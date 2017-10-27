import os
import json
from csv import DictReader, DictWriter

import re
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from numpy import array
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import *
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.metrics import accuracy_score
import sklearn.preprocessing as preprocessing



if __name__ == "__main__":

    # Read in data

    X_train = []
    Y_train = []
    X_test  = []
    Y_test  = []
    X_id = []
    i=0
    with open('train.csv',encoding="utf8") as f:
        header = f.readline()
        print(header)
        for data in f.readlines():    
            line = data.split(',')
            data_line = [int(line[1]),int(line[2]),int(line[3]),int(line[4]),int(line[1])/int(line[4]),int(line[2])/int(line[4])]
            X_train.append(data_line)
            Y_train.append(int(line[5]))
            i+=1
    with open('test.csv',encoding="utf8") as f:
        header = f.readline()
        print(header)
        for data in f.readlines():    
            line = data.split(',')
            data_line = [int(line[1]),int(line[2]),int(line[3]),int(line[4]),int(line[1])/int(line[4]),int(line[2])/int(line[4])]
            X_test.append(data_line)
            X_id.append(int(line[0]))
            i+=1
    print("total data read: ",i)
    X_train_norm = preprocessing.normalize(X_train, norm='l2')
    X_test_norm = preprocessing.normalize(X_test, norm='l2')
    print(X_train_norm[1],X_test_norm[1])
    print(X_train[1],X_test[1])
    
    lr = SGDClassifier(max_iter=1000,loss='log',alpha=0.0000001,penalty='l2',learning_rate='optimal',verbose=1,shuffle=True)
    lr.fit(X_train_norm, Y_train)
    y_pred = lr.predict_proba(X_test_norm)
    print(lr.classes_)
    i = 0
    c=0
    with open('test.csv', 'r') as src:
        with open('submission.csv', 'w') as dest:
            dest.write(src.readline())
            for line in src:
                dest.write('%s%s%s\n' % (line.split(',')[0],',',y_pred[i][1]))
                print(y_pred[i],X_test[i])
                if(y_pred[i][1] > 0.5):
                    c+=1
                i+=1
    print(c)
    
