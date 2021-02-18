import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix

import seaborn as sns


Dataset = pd.read_csv("Iris.csv")

previsores = Dataset.iloc[:, 1:5].values
classes = Dataset.iloc[:, 5].values

classes = LabelEncoder().fit_transform(classes)

previsores_treinamento, previsores_teste = train_test_split(previsores, test_size = 0.75, random_state = 0)
classe_treinamento, classe_teste = train_test_split(classes, test_size = 0.75, random_state = 0)

classificador = GaussianNB()

classificador.fit(previsores_treinamento, classe_treinamento)
previsoes = classificador.predict(previsores_teste)
precisao = accuracy_score(classe_teste, previsoes)
matriz = confusion_matrix(classe_teste, previsoes)

#sns.barplot(data = Dataset, x="SepalLengthCm", y="Species" )
sns.barplot(data = Dataset, x="PetalLengthCm", y="Species")
print("Precisão: {}%".format(precisao * 100))


