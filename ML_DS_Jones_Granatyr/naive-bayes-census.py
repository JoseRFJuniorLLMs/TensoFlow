import pandas as pd

base = pd.read_csv('census.csv')

previsores = base.iloc[:, 0:14].values
classe = base.iloc[:, 14].values
                
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_previsores = LabelEncoder()
previsores[:, 1] = labelencoder_previsores.fit_transform(previsores[:, 1])
previsores[:, 3] = labelencoder_previsores.fit_transform(previsores[:, 3])
previsores[:, 5] = labelencoder_previsores.fit_transform(previsores[:, 5])
previsores[:, 6] = labelencoder_previsores.fit_transform(previsores[:, 6])
previsores[:, 7] = labelencoder_previsores.fit_transform(previsores[:, 7])
previsores[:, 8] = labelencoder_previsores.fit_transform(previsores[:, 8])
previsores[:, 9] = labelencoder_previsores.fit_transform(previsores[:, 9])
previsores[:, 13] = labelencoder_previsores.fit_transform(previsores[:, 13])

onehotencoder = OneHotEncoder(categorical_features = [1,3,5,6,7,8,9,13])
previsores = onehotencoder.fit_transform(previsores).toarray()

labelencoder_classe = LabelEncoder()
classe = labelencoder_classe.fit_transform(classe)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
previsores = scaler.fit_transform(previsores)

from sklearn.cross_validation import train_test_split
previsores_treinamento, previsores_teste, classe_treinamento, classe_teste = train_test_split(previsores, classe, test_size=0.15, random_state=0)

from sklearn.naive_bayes import GaussianNB
classificador = GaussianNB()
classificador.fit(previsores_treinamento, classe_treinamento)
previsoes = classificador.predict(previsores_teste)

from sklearn.metrics import confusion_matrix, accuracy_score
precisao = accuracy_score(classe_teste, previsoes)
matriz = confusion_matrix(classe_teste, previsoes)

#Aqui estou passando os dados de treinamento para o algoritmo criar o modelo, ou seja, montar a tabela de probabilidades de
#cada combinação de variáveis previsoras + classe.
from sklearn.naive_bayes import GaussianNB
classificador = GaussianNB()
classificador.fit(previsores_treinamento, classe_treinamento)
#Nesse ponto estou passando os dados de teste para o algoritmo aplicar o modelo e prever os possíveis resultados, ou seja,
#o atributo 'classe' dos dados de teste.
previsoes = classificador.predict(previsores_teste)

#Os métodos abaixo servem para:
#- accuracy_score: dá o percentual de precisão do algoritmo, comparando os dados da classe de teste (que sabemos os valores de
#  antemão) com o que o algoritmo calculou (previsoes).
#- confusion_matrix: mostra em forma de matriz quanto de acerto e erro houve para cada valor possível da classe de teste.
from sklearn.metrics import accuracy_score, confusion_matrix
precisao = accuracy_score(classe_teste, previsoes)
matrizconfusao = confusion_matrix(classe_teste, previsoes)
