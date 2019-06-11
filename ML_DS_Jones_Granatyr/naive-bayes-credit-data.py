import pandas as pd
import numpy as np

base = pd.read_csv('credit_data.csv')

#Corrigir dados inválidos. Está atribuindo a média de idades para as idades que estão negativas.
base.loc[base.age < 0, 'age'] = base['age'][base.age > 0].mean()

#Separar as colunas entre previsoras e classe para trabalhar os dados separadamente
previsores = base.iloc[:,1:4].values
classe = base.iloc[:,4].values

#Corrigir dados faltante (NaN - Not a Number). A estratégia aqui está sendo usar a média (mean).
#O método fit_transform aplica a regra e transforma os dados, conforme regra especificada na segunda linha (preencher
#valores faltantes com a média).
from sklearn.impute import SimpleImputer
simpleimputer = SimpleImputer(missing_values=np.nan, strategy='mean')
#simpleimputer.fit(previsores[:, 1:4])
previsores[:, 1:4] = simpleimputer.fit_transform(previsores[:, 1:4])

#Colocar os atributos na mesma escala de valores (Escalonamento de Variáveis)
from sklearn.preprocessing import StandardScaler
standardscaler = StandardScaler()
previsores = standardscaler.fit_transform(previsores)

#Dividir a base em dados de treino e dados de teste
from sklearn.model_selection import train_test_split
previsores_treinamento, previsores_teste, classe_treinamento, classe_teste = train_test_split(previsores, classe, test_size=0.25, random_state=0)

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
