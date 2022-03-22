# Caso: Ganho Anual - PrevisaoML
Cenario do previsao de ganhos totais usando Machine Learning.

<b>Machine Learning nao se ve apenas no filme MATRIX.</b>

Neste caso de estudo, foi usado sklearn e pandas para a construcao de um algoritmo com Machine Learning que faz a previsao de ganho atraves de uma base de dados provinda de um arquivo CSV. De forma resumida, atraves dos dados percebe-se uma relacao entre alguns atributos das pessoas usadas no caso e seu ganho anual/mensal, atraves de calculos destes atributos, uma vez que um padrao e formado, e possivel prever com certa precisao qual sera a o ganho de pessoas uma vez inseridas os atributos necessarios. Para os mais metodicos, ha um arquivo JUPYTER NOTEBOOK incluido para melhor visualizacao.

Passo a passo do algoritmo explicado:

<h3>Download das dependencias</h3>
Foram utilizadas neste caso de estudo as bibliotecas pandas e sklearn:

``pip install pandas`` <br>
``pip install sklearn``

Apenas copie e cole o codigo acima no terminal com pip para instala-las.

<h3>Importacao dos pacotes e recursos necessarios </h3>

``import pandas as pd`` <br>
``from sklearn.preprocessing import StandardScaler``<br>
``from sklearn.preprocessing import LabelEncoder``<br>
``from sklearn.model_selection import train_test_split``<br>
``from sklearn.linear_model import LogisticRegression``<br>
``from sklearn.metrics import accuracy_score``<br>

<h3>Importacao da base de dados utilizada</h3>

``basedados = pd.read_csv('census.csv')`` <br>

Visualizaremos antes e depois da transformacao dos dados para melhor compreensao. <i>Note que temos 15 colunas (de 0 a 14 para o algoritmo), sendo que de 0 a 13 usaremos para prever a renda na coluna 14. </i><br>

``basedados.head()``<br>

![database](https://user-images.githubusercontent.com/23524569/159412934-8b0bcce6-c681-4426-9ba0-06793d8ed00b.png)


<h3>Dos Atributos e Metas/Resultado</h3>

Nossa base de dados sera dividida em 2 partes inicialmente (sera dividida novamente depois), partes dos dados as quais queremos tratar sao os atributos previsores, estes que usaremos para treinar o classificador/modelo, e outra sao as metas (ou resultados), os quais queremos prever apos modelo treinado. Antes de tudo, uma vez que temos no momento apenas 2 tipos possiveis na nossa base como atributo meta (' <=50K' e ' >50K') vamos categoriza-lo para o algoritmo possa retornar valores unicos.

``basedados['income'].unique()``

<h4> Separacao dos atributos previsores e do atributo meta</h4>

Neste momento, podemos dividir ainda mais nossa base de dados para posteriormente, preparar os dados para a criacao e treinamento do modelo. Sera agora, de fato, dividida a base em atributos previsores e metas.<br>
<i>Nota: para o algoritmo, : significa que pegamos todos os registros (linhas da base) da coluna 0 ATE (sem incluir) 14; o que nao acontece na linha de baixo, onde retiramos especificamente todos os registros da coluna 14.</i>

``attrPrevis = basedados.iloc[:, 0:14].values``<br>
``metas = basedados.iloc[:, 14].values``<br>

<h4>Preparacao dos dados nao numericos</h4>

No sklearn, as colunas com strings sao consideradas como variaveis categoricas, nao numericas, ou seja, variaveis com valores nominais e sem escala, a utilizacao dos dados dessa forma para o sklearn pode atrapalhar a etapa de aprendizagem, entao nosso proximo passo e trabalhar com um encoder para transformarmos todas as colunas categoricas em colunas com valores "matematicamente uteis" para o sklearn.

<i>Note que as colunas categoricas sao: workclass,	final-weight,	education,	marital-status,	occupation,	relationship,	race(etnia),	sex e	native-country; no momento, representados pelas indices 1, 3, 5, 6, 7, 8, 9 e 13</i>

``labelEncoder = LabelEncoder()`` <br>
``attrPrevis[:,1] = labelEncoder.fit_transform(attrPrevis[:,1])``<br>
``attrPrevis[:,3] = labelEncoder.fit_transform(attrPrevis[:,3])``<br>
``attrPrevis[:,5] = labelEncoder.fit_transform(attrPrevis[:,5])``<br>
``attrPrevis[:,6] = labelEncoder.fit_transform(attrPrevis[:,6])``<br>
``attrPrevis[:,7] = labelEncoder.fit_transform(attrPrevis[:,7])``<br>
``attrPrevis[:,8] = labelEncoder.fit_transform(attrPrevis[:,8])``<br>
``attrPrevis[:,9] = labelEncoder.fit_transform(attrPrevis[:,9])``<br>
``attrPrevis[:,13] = labelEncoder.fit_transform(attrPrevis[:,13])``<br>

Agora, mais uma vez, visualizaremos os dados para termos nocao real das mudancas feitas pelo encoder. Note que eles foram transformados pelo encoder de uma base de dados em CSV para uma array.

``attrPrevis.head()``<br>

![attrPrevis](https://user-images.githubusercontent.com/23524569/159412983-6b2b2b6f-2bc8-483a-8381-02f5de28de08.png)

<h4>Padronizacao dos dados</h4>

Na nossa base de dados, varios dos dados estao "relativamente" distantes entre si, temos ao mesmo tempo, numeros muito proximos uns dos outros quanto muito longe quando comparados a outros. Neste passo, no intuito de salvar recursos computacionais, faremos uma padronizacao da escala, que torna muito mais facil e vantojoso para nosso classificador/modelo treina-los posteriormente. 

``scalerAttrPrevis = StandardScaler()`` <br>
``attrPrevisScaled = scalerAttrPrevis.fit_transform(attrPrevis)`` <br>

![scaledattribprevis](https://user-images.githubusercontent.com/23524569/159413593-1bca2bb3-cd76-4ef7-8935-d6cd2464adec.png)


Se visualizarmos os dados agora, notaremos que estao muito proximos em relacao a escala utilizada, o que facilita tambem a visualizacao em graficos e plotagens.

<h3> Datasets (Conjunto de dados) e Classificadores </h3>

So entao (finalmente), apos os dados terem passado pelos estagios de importacao e preparacao, podemos comecar a definir os conjuntos da dados (datasets) que serao utilizados para fazermos os treinamentos e testes. Serao divididos em dois grupos: treinamento e teste; destes dois grupos, derivamos outros dois de cada um, treinamento/teste das metas e treinamento/teste dos atributos previsores. Felizmente, o sklearn ja possui funcoes para fazer esta divisao. 

<i>Nota: a forma e quantidade na qual sao dividas influencia diretamente nas previsoes e taxas de acerto do classificados/modelo </i>

``attrPrevis_train, attrPrevis_test, metas_train, metas_test = train_test_split(attrPrevisScaled, metas, test_size=0.3)`` <br>

Depois de devidamente agrupados, criamos o classificador e fazemos seu treinamento:

``classifier = LogisticRegression(max_iter = 10000)``<br>
``classifier.fit(attrPrevis_train, metas_train)``<br>

<h3> Fim: Previsoes e Verificacao da Taxa de Acerto </h3>

Uma vez treinado, podemos agora utilizar nosso classificador/modelo para fazer previsoes e verificarmos a taxa de acerto segundo os dados reais e os que o classificador gerou. Entao:

``previsaoFinal = classifier.predict(attrPrevis_test)`` <br>
``print('Previsao: ', previsaoFinal)``<br>

![predict](https://user-images.githubusercontent.com/23524569/159414854-dfa0f04a-e7d0-4568-9df8-42238bf1805a.png)

<i> Parte da previsao do nosso classificador (tambem uma array com o que o codigo "acredita" ser o resultado certo)</i>

E verificamos o quanto ele acertou segundo o treinamento feito:

``taxaAcerto = accuracy_score(metas_test, previsaoFinal)`` <br>
``print('Taxa de Acerto (%): ',taxaAcerto)`` <br>

![hit](https://user-images.githubusercontent.com/23524569/159414967-8f867fee-ae06-418e-9b22-e5886d9cdc18.png)

Podemos ver que nosso algoritmo teve uma taxa de acerto de 75.96%. 
