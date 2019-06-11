import pandas as pd
import numpy as np

#Ler o arquivo CSV
base = pd.read_csv('credit_data.csv')
#Corrigir dados inválidos. Está atribuindo a média de idades para as idades que estão negativas.
base.loc[base.age < 0, 'age'] = base['age'][base.age > 0].mean()

#Separar as colunas previsoras e classe em variáveis.
previsores = base.iloc[:, 1:4].values
classe = base.iloc[:, 4].values

#Corrigir dados faltante (NaN - Not a Number). A estratégia aqui está sendo usar a média (mean).
from sklearn.impute import SimpleImputer
simpleimputer = SimpleImputer(missing_values=np.nan, strategy='mean')
simpleimputer.fit(previsores[:, 1:4])
previsores[:, 1:4] = simpleimputer.transform(previsores[:, 1:4])

#Colocar os atributos na mesma escala (Escalonamento de Variáveis)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
previsores = scaler.fit_transform(previsores)

#Dividir a base em dados de treino e dados de teste
from sklearn.model_selection import train_test_split
previsores_treinamento, previsores_teste, classe_treinamento, classe_teste = train_test_split(previsores, classe, test_size=0.25, random_state=0)

# importação da biblioteca
# criação do classificador
from sklearn.naive_bayes import GaussianNB
classificador = GaussianNB()
classificador.fit(previsores_treinamento, classe_treinamento)
#previsoes = classificador.predict(previsores_teste)

#from sklearn.metrics import confusion_matrix, accuracy_score
#precisao = accuracy_score(classe_teste, previsoes)
#matriz = confusion_matrix(classe_teste, previsoes)

print('')