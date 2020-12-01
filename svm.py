from getEmbeddings import getEmbeddings
import numpy as np
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import scikitplot.plotters as skplt



def plot_cmat(yte, ypred):
    '''Plotting confusion matrix'''
    skplt.metrics.plot_confusion_matrix(yte,ypred)
    plt.show()


xtr,xte,ytr,yte = getEmbeddings("datasets/train.csv")
np.save('./xtr', xtr)
np.save('./xte', xte)
np.save('./ytr', ytr)
np.save('./yte', yte)





xtr = np.load('./xtr.npy')
xte = np.load('./xte.npy')
ytr = np.load('./ytr.npy')
yte = np.load('./yte.npy')

clf = SVC()

clf.fit(xtr, ytr)

y_pred = clf.predict(xte)
m = yte.shape[0]
n = (yte != y_pred).sum()


plot_cmat(yte, y_pred)

print("Accuracy = " + format((m-n)/m*100, '.2f') + "%")   # 88.42%
from sklearn.metrics import classification_report
target_names = ['class 0', 'class 1']
print(classification_report(yte, y_pred, target_names=target_names))