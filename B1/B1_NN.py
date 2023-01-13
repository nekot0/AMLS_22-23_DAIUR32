import numpy as np
import matplotlib.pyplot as pp
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

import time


### Parameters ###
INPUT_FEATURES = 19
LAYER1_NEURONS = 2048
LAYER2_NEURONS = 1024
LAYER3_NEURONS = 512
OUTPUT_RESULTS = 5

BATCH_SIZE = 15
LEARNING_RATE = 1e-5
EPOCH = 30

global model, optimizer, criterion


### Neural Network Class

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()

        #self.flatten = nn.Flatten()

        self.layer1 = nn.Linear(INPUT_FEATURES, LAYER1_NEURONS)
        self.layer2 = nn.Linear(LAYER1_NEURONS, LAYER2_NEURONS)
        self.layer3 = nn.Linear(LAYER2_NEURONS, LAYER3_NEURONS)
        self.layer_out = nn.Linear(LAYER3_NEURONS, OUTPUT_RESULTS)

        self.act1 = nn.ReLU()
        self.act2 = nn.ReLU()
        self.act3 = nn.ReLU()

    def forward(self, x):
        #x = self.flatten(x)
        x = self.act1(self.layer1(x))
        x = self.act2(self.layer2(x))
        x = self.act3(self.layer3(x))
        x = self.layer_out(x)
        return x


### Neural Network fit & evaluation

def init_parameters(layer):
    if type(layer) == nn.Linear:
        nn.init.normal_(layer.weight, std=1)
        nn.init.normal_(layer.bias, std=1)


def B1_NN(Xtrain, Ytrain, Xtest, Ytest):
    model = NeuralNetwork()
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, weight_decay=0)
    criterion = nn.CrossEntropyLoss()

    # Initialise parameters
    model.apply(init_parameters)

    avg_loss = 0.0
    cost = []
    train_data = TensorDataset(Xtrain, Ytrain)
    loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)

    # Fitting for training data
    start = time.time()
    model.train()

    for epoch in tqdm(range(EPOCH)):
        total_loss = 0.0

        for train_X, train_Y in loader:
            pred = model(train_X)
            optimizer.zero_grad()
            loss = criterion(pred, train_Y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        n = epoch + 1
        avg_loss = total_loss/n
        cost.append(avg_loss)
        
    end = time.time()
    print('Fitting end: time ', f'{end-start:.1f}', 'sec')


    # Evaluation for training data
    pred = nn.Softmax(dim=1)(model(Xtrain)).argmax(1)
    Yindex = Ytrain.argmax(1)
    
    accurate = 0
    for i in range(pred.shape[0]):
        if pred[i] == Yindex[i]:
            accurate += 1
    print('Accuracy for training data:', f'{(accurate*100)/pred.shape[0]:.1f}', '%')


    # Evaluation for test data
    pred = nn.Softmax(dim=1)(model(Xtest)).argmax(1)
    Yindex = Ytest.argmax(1)

    accurate = 0

    for i in range(pred.shape[0]):
        if pred[i] == Yindex[i]:
            accurate += 1
    print('Accuracy for test data:', f'{(accurate*100)/pred.shape[0]:.1f}', '%')

    

