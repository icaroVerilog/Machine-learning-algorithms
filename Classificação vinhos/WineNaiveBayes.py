#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 22 10:06:51 2020

@author: icaro
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder

database = pd.read_csv("wine_dataset.csv")

previsores = database.iloc[:, 0:12].values
classe = database.iloc[:, 12].values

labelencorder_classe = LabelEncoder()
classe = labelencorder_classe.fit_transform(classe)

previsores_treinamento, previsores_teste, classe_treinamento, classe_teste = train_test_split(previsores, classe, test_size=0.25, random_state=0)

classificador = GaussianNB()
classificador.fit(previsores_treinamento, classe_treinamento)

previsoes = classificador.predict(previsores_teste)
precisao = accuracy_score(classe_teste, previsoes)
matriz = confusion_matrix(classe_teste, previsoes)

print("Precisão: {}%".format(precisao * 100))