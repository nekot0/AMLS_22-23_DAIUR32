import numpy as np
import matplotlib.pyplot as pp
from tqdm import tqdm
import time

from sklearn import svm
from sklearn.metrics import accuracy_score


def A1_SVM(Xtrain, Ytrain, Xtest, Ytest):
    model = svm.SVC(kernel='rbf', gamma=0.1, C=100.0)

    Xtrain_in = Xtrain.reshape((Xtrain.shape[0], Xtrain.shape[1]*Xtrain.shape[2]))
    Ytrain_in = Ytrain[:,1]
    Xtest_in = Xtest.reshape((Xtest.shape[0], Xtest.shape[1]*Xtest.shape[2]))
    Ytest_in = Ytest[:,1]

    start = time.time()
    model.fit(Xtrain_in, Ytrain_in)
    end = time.time()
    print('Fitting end: time ', f'{end-start:.1f}', 'sec')

    pred = model.predict(Xtrain_in)
    print('Accuracy for training data:', f'{100*accuracy_score(pred, Ytrain_in):.1f}', '%')

    pred = model.predict(Xtest_in)
    print('Accuracy for test data:', f'{100*accuracy_score(pred, Ytest_in):.1f}', '%')
