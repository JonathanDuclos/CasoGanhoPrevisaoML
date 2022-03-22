import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

#Importacao dos dados e visualizacao
basedados = pd.read_csv('census.csv')
basedados.head()

#Categorizado a coluna INCOME
basedados['income'].unique()

#Separacao dos atributos
attrPrevis = basedados.iloc[:, 0:14].values
metas = basedados.iloc[:, 14].values

#Transformando as classes categoricas em informacao matematicamente util
labelEncoder = LabelEncoder()
attrPrevis[:,1] = labelEncoder.fit_transform(attrPrevis[:,1])
attrPrevis[:,3] = labelEncoder.fit_transform(attrPrevis[:,3])
attrPrevis[:,5] = labelEncoder.fit_transform(attrPrevis[:,5])
attrPrevis[:,6] = labelEncoder.fit_transform(attrPrevis[:,6])
attrPrevis[:,7] = labelEncoder.fit_transform(attrPrevis[:,7])
attrPrevis[:,8] = labelEncoder.fit_transform(attrPrevis[:,8])
attrPrevis[:,9] = labelEncoder.fit_transform(attrPrevis[:,9])
attrPrevis[:,13] = labelEncoder.fit_transform(attrPrevis[:,13])

#Aplicando uma padronizacao de escala
scalerAttrPrevis = StandardScaler()
attrPrevisScaled = scalerAttrPrevis.fit_transform(attrPrevis)

#Preparando os dados para agrupamento e definicao dos datasets
attrPrevis_train, attrPrevis_test, metas_train, metas_test = train_test_split(attrPrevisScaled, metas, test_size=0.3)

#Criando o classificador e fazendo o treinamento
classifier = LogisticRegression(max_iter = 10000)
classifier.fit(attrPrevis_train, metas_train)

#Verificacao do classificador
previsaoFinal = classifier.predict(attrPrevis_test)
print('Previsao: ', previsaoFinal)

#Analisando a taxa de acerto do classificador
taxaAcerto = accuracy_score(metas_test, previsaoFinal)
print('Taxa de Acerto (%): ',taxaAcerto)

