###### Import Libraries ######
import numpy as np
import matplotlib.pyplot as pp
import math
from tqdm import tqdm

### Classifiers
# Neural Network
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
# SVM
from sklearn import svm
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score
# AdaBoost
from sklearn.ensemble import AdaBoostClassifier

### Functions for tasks
import A1.load_celeba as load_celeba    # Load data & extract 68 facial landmarks
import A1.A1_NN as A1_NN   # Neural Network model for task A1
import A1.A1_SVM as A1_SVM  #SVM model for A1
import A1.A1_AdaBoost as A1_AdaBoost    #AdaBoost model for A1

import A2.A2_NN as A2_NN    # Neural Network model for task A2
import A2.A2_SVM as A2_SVM  # SVM model for A2
import A2.A2_AdaBoost as A2_AdaBoost    # AdaBoost model for A2

import B1.load_cartoon as load_cartoon  #Load data & extract 81 facial landmarks & calculate 19 features + skin&hair colours
import B1.B1_SVM as B1_SVM  # SVM model for B1
import B1.B1_NN as B1_NN    # Neural Network model for task B1

import B2.B2_SVM as B2_SVM  # SVM model for B2
import B2.B2_NN as B2_NN    # Neural Network model for task B2



###### Celeba data load ######
print('\n')
print('###### Start loading celeba & celeba_test ######\n')

# Load data & extract HoG features
print('---------------')
print('Loading training data & extracting 68 landmarks...')
Xtrain, Y1train, Y2train, undetected_files, total_time = load_celeba.get_data(mode='celeba-train')
print('HoG features loaded')
print('Successfully extracted:', f'{Xtrain.shape[0]*100/(Xtrain.shape[0]+len(undetected_files)):.1f}', '%')
print('Calculation time:', f'{total_time:.1f}', 'sec')

print('\n')
print('Loading test data & extracting 68 landmarks...')
Xtest, Y1test, Y2test, undetected_files, total_time = load_celeba.get_data(mode='celeba-test')
print('HoG features loaded')
print('Successfully extracted:', f'{Xtest.shape[0]*100/(Xtest.shape[0]+len(undetected_files)):.1f}', '%')
print('Calculation time:', f'{total_time:.1f}', 'sec')
print('---------------\n')

# Normalising data
Xtrain = torch.from_numpy(Xtrain).float()
Y1train = torch.from_numpy(Y1train).float()
Y2train = torch.from_numpy(Y2train).float()
Xtest = torch.from_numpy(Xtest).float()
Y1test = torch.from_numpy(Y1test).float()
Y2test = torch.from_numpy(Y2test).float()
Xtrain = (Xtrain-Xtrain.mean()) / np.sqrt(Xtrain.var())
Xtest = (Xtest-Xtest.mean()) / np.sqrt(Xtest.var())



###### A1 Code ######
print('\n')
print('###### Start task A1 ######\n')

# Neural Network model
print('Train & Evaluate by Neural Network model...')
A1_NN.A1_NN(Xtrain, Y1train, Xtest, Y1test)
print('\n')

# SVM model
print('Train & Evaluate by SVM model...')
A1_SVM.A1_SVM(Xtrain, Y1train, Xtest, Y1test)
print('\n')

# AdaBoost model
print('Train & Evaluate by AdaBoost model...')
A1_AdaBoost.A1_AdaBoost(Xtrain, Y1train, Xtest, Y1test)
print('---------------\n')

print('###### End A1 ######\n')

#### A1 Code End ####



###### A2 Code ######
print('\n')
print('###### Start task A2 ######\n')

# Neural Network model
print('Train & Evaluate by Neural Network model...')
A2_NN.A2_NN(Xtrain, Y2train, Xtest, Y2test)
print('\n')

# SVM model
print('Train & Evaluate by SVM model...')
A2_SVM.A2_SVM(Xtrain, Y2train, Xtest, Y2test)
print('\n')

# AdaBoost model
print('Train & Evaluate by AdaBoost model...')
A2_AdaBoost.A2_AdaBoost(Xtrain, Y2train, Xtest, Y2test)
print('---------------\n')

print('###### End A2 ######\n')

#### A2 Code End ####



###### Cartoon data load ######
print('\n')
print('###### Start loading cartoon_set & cartoon_set_test ######\n')

# Load data & extract HoG features
print('---------------')
print('Loading training data & calculating features using 81 landmarks...')
Xtrain, Y1train, Y2train, undetected_files, total_time = load_cartoon.get_data(mode='cartoon-train')
print('HoG features loaded')
print('Successfully extracted:', f'{Xtrain.shape[0]*100/(Xtrain.shape[0]+len(undetected_files)):.1f}', '%')
print('Calculation time:', f'{total_time:.1f}', 'sec')

print('\n')
print('Loading test data & calculating features using 81 landmarks...')
Xtest, Y1test, Y2test, undetected_files, total_time = load_cartoon.get_data(mode='cartoon-test')
print('HoG features loaded')
print('Successfully extracted:', f'{Xtest.shape[0]*100/(Xtest.shape[0]+len(undetected_files)):.1f}', '%')
print('Calculation time:', f'{total_time:.1f}', 'sec')
print('---------------\n')

# Normalising data
Xtrain = torch.from_numpy(Xtrain).float()
Y1train = torch.from_numpy(Y1train).float()
Y2train = torch.from_numpy(Y2train).float()
Xtest = torch.from_numpy(Xtest).float()
Y1test = torch.from_numpy(Y1test).float()
Y2test = torch.from_numpy(Y2test).float()
Xtrain = (Xtrain-Xtrain.mean()) / np.sqrt(Xtrain.var())
Xtest = (Xtest-Xtest.mean()) / np.sqrt(Xtest.var())
for i in range(Xtrain.shape[1]):
    Xtrain[:,i] = (Xtrain[:,i]-Xtrain[:,i].mean())/np.sqrt(Xtrain[:,i].var())
    Xtest[:,i] = (Xtest[:,i]-Xtest[:,i].mean())/np.sqrt(Xtest[:,i].var())
X1train = Xtrain[:,0:19]
X1test = Xtest[:,0:19]



###### B1 Code ######
print('\n')
print('###### Start task B1 ######\n')

# SVM model
print('Train & Evaluate by SVM model...')
B1_SVM.B1_SVM(X1train, Y1train, X1test, Y1test)
print('\n')

# Neural Network model
print('Train & Evaluate by Neural Network model...')
B1_NN.B1_NN(X1train, Y1train, X1test, Y1test)
print('\n')

print('###### End B1 ######\n')

#### B1 Code End ####



###### B2 Code ######
print('\n')
print('###### Start task B2 ######\n')

# SVM model
print('Train & Evaluate by SVM model...')
B2_SVM.B2_SVM(Xtrain, Y2train, Xtest, Y2test)
print('\n')

# Neural Network model
print('Train & Evaluate by Neural Network model...')
B2_NN.B2_NN(Xtrain, Y2train, Xtest, Y2test)
print('\n')

print('###### End B2 ######\n')

#### B2 Code End ####

print('###### End all the tasks ######')