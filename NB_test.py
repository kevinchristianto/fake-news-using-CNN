# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 12:43:01 2020

@author: manoj
"""
from getEmbeddings import getEmbeddings
from sklearn.naive_bayes import GaussianNB
import numpy as np
import matplotlib.pyplot as plt
import scikitplot.plotters as skplt

xtr = np.load('./xtr.npy')
xte = np.load('./xte.npy')
ytr = np.load('./ytr.npy')
yte = np.load('./yte.npy')

gnb = GaussianNB()
gnb.fit(xtr,ytr)
y_pred = gnb.predict(xte)
m = yte.shape[0]
n = (yte != y_pred).sum()
#print("Accuracy = " + format((m-n)/m*100, '.2f') + "%")   # 72.94%

plot_cmat(yte, y_pred)


print("Accuracy = " + format((m-n)/m*100, '.2f') + "%")   # 88.42%
from sklearn.metrics import classification_report
target_names = ['class 0', 'class 1']
print(classification_report(yte, y_pred, target_names=target_names))