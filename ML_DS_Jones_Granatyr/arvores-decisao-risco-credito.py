import pandas as pd

base = pd.read_csv('risco_credito.csv')

#Dividir os atributos (campos) em previsores e classe (classe é o que eu quero prever)
previsores = base.iloc[:,0:4].values
classe = base.iloc[:,4].values

#Uso do LabelEncoder para transformar variáveis categóricas em numéricas discretas.
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
previsores[:,0] = labelencoder.fit_transform(previsores[:,0])
previsores[:,1] = labelencoder.fit_transform(previsores[:,1])
previsores[:,2] = labelencoder.fit_transform(previsores[:,2])
previsores[:,3] = labelencoder.fit_transform(previsores[:,3])

#Utilisado o algoritmo de árvore DecisionTreeClassifier. Usado o parâmetro entropy e não o gini, para ficar de
#acordo com o que foi passado na aula. A classe export será utilizada para visualizar a árvore.
from sklearn.tree import DecisionTreeClassifier, export
classificador = DecisionTreeClassifier(criterion='entropy')
classificador.fit(previsores, classe) #O fit é o método que cria o modelo, ou seja, gera a árvore.

#O atributo 'feature_importances'_ é bem interessante, pois apresenta a relevância de cada um dos atributos
#previsores, conforme aula 36 da sessão 5 e resumo sobre Montagem da Árvore.
print(classificador.feature_importances_)

#O método export_graphviz gera um arguivo que pode ser importado em algumas ferramentas para visualização
#gráfica da árvore. Pode-se visualizar na Web (http://www.webgraphviz.com/) ou instalar o graphviz.
export.export_graphviz(classificador,
                       out_file='arvore.dot',
                       feature_names=['historia','divida','garantias','renda'],
                       class_names=['alto', 'moderado', 'baixo'],
                       filled=True,
                       leaves_parallel=True)

# história boa, dívida alta, garantias nenhuma, renda > 35
# história ruim, dívida alta, garantias adequada, renda < 15
resultado = classificador.predict([[0,0,1,2], [3, 0, 0, 0]])
