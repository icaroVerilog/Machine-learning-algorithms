import numpy as NP
import pandas as PD
import torch
import torch.nn.functional as FNC
import matplotlib.pyplot as PLT
import torch.utils.data as data_utils # import da função para criar o trainset

from time import time
from torch import nn, optim
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split


dataset = PD.read_csv("heart_failure_clinical_records_dataset.csv")
b = dataset.drop("time", axis = 1)

def toTensor(dataframe):
    
    trainDataframe, validationDataframe = train_test_split(dataframe, test_size = 0.25, random_state = 1)
    
    # Criando um tensor do tipo inteiro 32 bits com a coluna alvo 
    trainTarget = torch.tensor(trainDataframe["DEATH_EVENT"].values)
    validationTarget = torch.tensor(validationDataframe["DEATH_EVENT"].values)
    
    # Criando um tensor do tipo inteiro 32 bits com 'os preditores
    train1 = torch.tensor(trainDataframe.drop("DEATH_EVENT", axis = 1).values)
    validation = torch.tensor(validationDataframe.drop("DEATH_EVENT", axis = 1).values)
    
    # Criando o tensor de treino
    trainTensor = data_utils.TensorDataset(train1, trainTarget)
    validationTensor = data_utils.TensorDataset(validation, validationTarget)
    
    # Criando o dataloader para inserir o tensor no modelo 
    trainLoader = data_utils.DataLoader(dataset = trainTensor, batch_size = 30, shuffle = True)
    validationLoader = data_utils.DataLoader(dataset = validationTensor, batch_size = 30, shuffle = True)
    
    return trainLoader, validationLoader, dataframe
    
trainLoader, validationLoader, A = toTensor(b)
        

# Estrutura da rede neural

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        
        # Definição das camadas
        
        self.linear1 = nn.Linear(11, 100) # Camada de entrada
        self.linear2 = nn.Linear(100, 75)
        self.linear3 = nn.Linear(75, 2)
        
    def forward(self, data):
        
        data = FNC.relu(self.linear1(data))
        data = FNC.relu(self.linear2(data))
        
        data = self.linear3(data)
        
        return FNC.sigmoid(data)
        
    def Train(self, model, trainLoader, device):
        
        model.double() # Resolve o problema https://github.com/pytorch/pytorch/issues/2138
        
        # Politica de pesos
        
        optmizer = optim.SGD(model.parameters(), lr = 0.001, momentum = 0.5)
        #optmizer = optim.Adam(model.parameters(), lr = 0.001, betas = (0.9, 0.999), eps = 1e-08, weight_decay = 0, amsgrad = False)
        
        begin = time()
        
        lossCritery = nn.NLLLoss()
        
        EPOCHS = 2000
        model.train()
        
        for epoch in range(EPOCHS): 
            acumulatedLoss = 0
            for train1, trainTarget in trainLoader:
                
                
                optmizer.zero_grad()
                
                output = model(train1.to(device))
                
                instantlyLoss = lossCritery(output, trainTarget.to(device))
                instantlyLoss.backward()
                optmizer.step()
                
                
                acumulatedLoss = acumulatedLoss + instantlyLoss.item()
                totalLoss = acumulatedLoss / len(trainLoader)
                
            else:
                
                print("Epoch {}/{} - Perda: {} - Acuracia: N/A".format(epoch + 1, EPOCHS, totalLoss))
                print("Tempo de treino (em minutos) =",(time() - begin) / 60)
                
"""
    def Validation(self, model, validationLoader, device):
        
        arrayPredictedValues = []
        arrayCorrectValues = []
        
        for validation, validationTarget in validationLoader:
            for i in range(len(validationTarget)):
""" 

model = Network()
device = torch.device("cpu") 
model.to(device)
model.Train(model, trainLoader, device)             
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    