import pandas as pd

#Leitura arquivo
base = pd.read_csv('credit_data.csv')
#Mostra estatísticas do arquivo importado
base.describe()

#TRATAMENTO DE VALORES INVÁLIDOS

#Duas maneiras de ler os registros cujo campo idade é menor que zero
base.loc[base['age'] < 0]
base.loc[base.age < 0]

#Achar a média da idade dos registros cuja idade é maior que zero e gravar
#essa média nos registros cuja idade está negativa.
base['age'][base.age > 0].mean()
base.loc[base.age < 0, 'age'] = base['age'][base.age > 0].mean()

#TRATAMENTO DE VALORES FALTANTES
#ler os registros cujo campo idade está nulo
base.loc[pd.isnull(base['age'])]

#Antes de fazer o tratamento dos dados é importante separar os dados do
#dataframe em previsores e o atributo classe (classe a qual pertence cada registro).
#É dessa maneira que o scikit-learn e outras bibliotecas se comportam/trabalham.
#Assim preservamos o dataframe.

#O método abaixo serve para atribuir valores onde eles forem nulos
from sklearn.preprocessing import Imputer
previsores = base.iloc[:, 1:4].values
classe = base.iloc[:, 4].values

imputer = Imputer(missing_values='NaN', strategy='mean',axis=0)
imputer = imputer.fit(previsores[:, 0:3])
previsores[:, 0:3] = imputer.transform(previsores[:, 0:3])
