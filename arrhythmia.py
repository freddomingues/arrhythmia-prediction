"""
@author: fred_
"""

import numpy as np
import pandas as pd
base = pd.read_csv('arrhythmia.csv')
previsores = base.iloc[:, 0:279].values
classe = base.iloc[:,279].values

#corrigindo valores faltantes   
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='constant') 
imputer = imputer.fit(previsores[:,0:279])
previsores[:,0:279] = imputer.transform(previsores[:,0:279])

import matplotlib.pyplot as plt
plt.scatter(previsores[:,0], previsores[:,3], c=classe, s=30, alpha=1, edgecolors='k')
plt.title('Idade x Peso para classificação de cardiopatias')
plt.xlabel('Idade')
plt.ylabel('Peso')
cbar = plt.colorbar()
cbar.set_ticks([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16])
cbar.set_ticklabels(['Normal','Doença da Artéria Coronária',
                         'Infarto do Miocárdio Anterior','Infarto do Miocárdio Inferior',
                         'Taquicardia Sinusal','Braquicardia Sinusal',
                         'Contração Ventricular Prematura','Contração Supraventricular Prematuraa',
                         'Bloqueio de Ramo Esquerdo','Bloco de Ramo Direito',
                         '1 Grau de Bloqueio Atrioventricular','2 Grau de Bloqueio AV',
                         '3 Grau de Bloqueio AV','Hipertrofia Ventricular Esquerda',
                         'Fibrilação Atrial','Outras'])
plt.show()

#escalonamento dos dados
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
previsores = scaler.fit_transform(previsores)

#--------------------Regressão Logística------------------------
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
kfold = StratifiedKFold(n_splits = 10, shuffle=True, random_state = 5)
resultadosRL = []
palpitesRL = []
realRL = []
for indice_treinamento, indice_teste in kfold.split(previsores, np.zeros(shape=[previsores.shape[0],1])):
    classificador = LogisticRegression()
    classificador.fit(previsores[indice_treinamento], classe[indice_treinamento])
    previsoesRL = classificador.predict(previsores[indice_teste])
    precisao = accuracy_score(classe[indice_teste], previsoesRL)
    palpitesRL.append(previsoesRL)
    realRL.append(classe[indice_teste])    
    resultadosRL.append(precisao)

resultadosRL = np.asarray(resultadosRL)
mediaRL = resultadosRL.mean()

pRL =[]
rRL = []
for palpite in palpitesRL:
    for pal in palpite:
        pRL.append(pal)

for realidade in realRL:
    for reali in realidade:
        rRL.append(reali)

import matplotlib.pyplot as plt
x = [1,16]
y = [1,16]
plt.scatter(rRL, pRL, alpha=1, edgecolors='k')
plt.plot(x,y, color="green")
plt.title('Previsões com Regressão Logística x Realidade')
plt.xlabel('Realidade')
plt.ylabel('Previsões')
plt.show()

#--------------------Random Forest-------------------------------
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
kfold = StratifiedKFold(n_splits = 10, shuffle=True, random_state = 5)
resultadosRF = []
palpitesRF = []
realRF = []
for indice_treinamento, indice_teste in kfold.split(previsores, np.zeros(shape=[previsores.shape[0],1])):
    classificador = RandomForestClassifier(n_estimators=50, criterion='entropy', random_state=0)
    classificador.fit(previsores[indice_treinamento], classe[indice_treinamento])
    previsoesRF = classificador.predict(previsores[indice_teste])
    precisao = accuracy_score(classe[indice_teste], previsoesRF)
    palpitesRF.append(previsoesRF)
    realRF.append(classe[indice_teste])    
    resultadosRF.append(precisao)

resultadosRF = np.asarray(resultadosRF)
mediaRF = resultadosRF.mean()

pRF =[]
rRF = []

for palpite in palpitesRF:
    for pal in palpite:
        pRF.append(pal)

for realidade in realRF:
    for reali in realidade:
        rRF.append(reali)

import matplotlib.pyplot as plt
x = [1,16]
y = [1,16]
plt.scatter(rRF, pRF, alpha=1, edgecolors='k')
plt.plot(x,y, color="green")
plt.title('Previsões com Random Forest x Realidade')
plt.xlabel('Realidade')
plt.ylabel('Previsões')
plt.show()
