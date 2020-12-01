# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 12:42:06 2020

@author: manoj
"""
from getEmbeddings import getEmbeddings
import numpy as np
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import scikitplot.plotters as skplt
import sklearn.metrics as metrics
def plot_cmat(yte, ypred):
    '''Plotting confusion matrix'''
    skplt.plot_confusion_matrix(yte,ypred)
    plt.show()

xtr = np.load('./xtr.npy')
xte = np.load('./xte.npy')
ytr = np.load('./ytr.npy')
yte = np.load('./yte.npy')

clf = SVC(probability=True)
clf.fit(xtr, ytr)

y_pred = clf.predict(xte)
m = yte.shape[0]
n = (yte != y_pred).sum()


plot_cmat(yte, y_pred)

print("Accuracy = " + format((m-n)/m*100, '.2f') + "%")   # 88.42%
from sklearn.metrics import classification_report
target_names = ['class 0', 'class 1']
print(classification_report(yte, y_pred, target_names=target_names))

probs = clf.predict_proba(xte)
preds = probs[:,1]
fpr, tpr, threshold = metrics.roc_curve(yte, preds)
roc_auc = metrics.auc(fpr, tpr)

plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()