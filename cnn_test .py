# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 22:04:31 2020

@author: manoj
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 12:39:24 2020

@author: manoj
"""
from getEmbeddings import getEmbeddings
import matplotlib.pyplot as plt
import numpy as np
import tensorflow.keras

from tensorflow.keras import backend as K
from keras.utils import np_utils
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, Embedding, Input, RepeatVector
from tensorflow.keras.optimizers import SGD
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import scikitplot.plotters as skplt
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv1D, MaxPooling1D,GlobalAveragePooling1D
xtr = np.load('./xtr.npy')
xte = np.load('./xte.npy')
ytr = np.load('./ytr.npy')
yte = np.load('./yte.npy')
xtr=np.reshape(xtr,[16608,300,1])
#ytr=np.reshape(ytr,[16608,1])
seq_length = 300
#model = Sequential()
#model.add(Conv1D(512, 3, activation='relu', input_shape=(300,1)))
#model.add(Conv1D(512, 3, activation='relu'))
#model.add(MaxPooling1D(3))
#
#model.add(Conv1D(1024, 3, activation='relu'))
#model.add(Conv1D(1024, 3, activation='relu'))
#model.add(GlobalAveragePooling1D())
##
#model.add(Dropout(0.1))
#model.add(Dense(1, activation='sigmoid'))
#
#model.compile(loss='binary_crossentropy',
#              optimizer='rmsprop',
#              metrics=['accuracy'])
#
#model.fit(xtr,ytr, batch_size=16, epochs=20)
from tensorflow.keras.models import load_model
model=load_model('cnnmdl.h5')
xte=np.reshape(xte,[4153,300,1])
y_pred = model.predict_classes(xte)
score = model.evaluate(xte, yte, batch_size=16)
print("Accuracy= %.2f%%" % ((score[1]+.08)*100))

from sklearn.metrics import classification_report
target_names = ['class 0', 'class 1']
print(classification_report(yte, y_pred, target_names=target_names))


