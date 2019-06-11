# -*- coding: utf-8 -*-
"""
Created on Sat Oct 20 09:02:45 2018

@author: Jean
"""

import pandas as pd

#Leitura da base de dados
base = pd.read_csv('risco_credito.csv')

#Separação da base entre atributos previsores e a classe
previsores = base.iloc[:,0:4].values
classe = base.iloc[:,4].values

#Transformaremos as variáveis categóricas em numéricas discretas, pois esse algoritmo não está preparado para tratar
#variáveis categóricas nos previsores.
from sklearn.preprocessing import LabelEncoder

#O LaberEncoder é uma classe que serve para transformar atributos categóricos em numéricos discretos.
labelencoder = LabelEncoder()
#No caso abaixo, estou pegando todas as linhas das colunas 0, 1, 2 e 3 e transformando o conteúdo categórico em numérico.
previsores[:,0] = labelencoder.fit_transform(previsores[:,0])
previsores[:,1] = labelencoder.fit_transform(previsores[:,1])
previsores[:,2] = labelencoder.fit_transform(previsores[:,2])
previsores[:,3] = labelencoder.fit_transform(previsores[:,3])

#Abaixo estou aplicando o algoritmo GaussianNB e gerando o modelo (método 'fit')
#Se aplicarmos direto, sem o código acima, vai apresentar o erro ValueError: could not convert string to float: 'acima_35'
#Isso ocorre porque esse algoritmo não está preparado para tratar variáveis categóricas (pelo menos não nos previsores).
from sklearn.naive_bayes import GaussianNB
classificador = GaussianNB()
classificador.fit(previsores, classe)

#Abaixo vamos fazer um teste com dois novos registros. Como não apresentou erro de conversão de string para a classe,
#significa que o algoritmo está preparado para tratar classes categóricas.
# história boa, dívida alta, garantias nenhuma, renda > 35
# história ruim, dívida alta, garantias adequada, renda < 15
#O método 'predict' serve para aplicar o modelo acima a novos valores.
resultado = classificador.predict([[0,0,1,2], [3, 0, 0, 0]])

print(classificador.classes_)
print(classificador.class_count_)
print(classificador.class_prior_)
print('')