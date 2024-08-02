'''
Modello di classificazione multiclasse con tecnica One vs All, utilizza più classificazioni binarie.
Utilizzo dataset di numeri scritti a mano
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, accuracy_score
#Dataset preset di sklearn
from sklearn.datasets import load_digits

#Carico dataset e creo arrays
digits = load_digits()
X = digits.data
Y = digits.target
#==> Il dataset ha 1797 esempi e 64 proprietà, ogni prop. rappresenta un pixel dell'immagine.

#Proviamo a visualizzare un'immagine partendo dalla sua matrice di pixel

for i in range(0,10):
    pic_matrix = X[Y==i][0].reshape([8,8])
    plt.imshow(pic_matrix, cmap="gray")
    plt.show()

    
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.3, random_state=0)
#Essendo delle immagini conviene normalizzare tutti i dati per facilitare addestramento
from sklearn.preprocessing import MinMaxScaler
mms = MinMaxScaler()
X_train = mms.fit_transform(X_train)
X_test = mms.transform(X_test)

#Passiamo ora alla classificazione multiclasse
#LogisticRegression riconosce automaticamente il multiclasse e applica la tecnica OneVSAll
lr = LogisticRegression()
lr.fit(X_train, Y_train)
Y_pred = lr.predict(X_test)
Y_pred_proba = lr.predict_proba(X_test)

#Calcolo metriche
print("ACCURACY= "+str(accuracy_score(Y_test, Y_pred)))
print("LOSS= "+str(log_loss(Y_test, Y_pred_proba)))

#Introduciamo nuova metrica, matrice di confusione, per vedere su quali classi il modello ha fatto più errori
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, Y_pred)

#Passiamo alla visualizzazione della matrice generata
import seaborn as sns
'''
Utilizzo una heatmap che indica per ciascuna classe quali e quanti errori ha fatto
Es. il valore 2 nella settima riga e quarta colonna, ci dice che 2 immagini rappresentanti un 8 son state
classificate come un 5! (conteggio parte sempre da zero)
'''
plt.figure(figsize=(9,9))
sns.heatmap(cm, annot=True, cmap="Blues_r", linewidths=.5, square=True)
plt.ylabel("Classe corrente")
plt.xlabel("Classe predetta")

#Sklearn implementa anche un altro modello di reg. multiclasse
from sklearn.multiclass import OneVsRestClassifier
#Quando lo si istanzia bisogna fornire un classificatore di base da utilizzare
ovr = OneVsRestClassifier(LogisticRegression())
#Dopodiché si utilizza come un modello qualunque
ovr.fit(X_train,Y_train)
Y_pred1 = ovr.predict(X_test)
Y_pred1_proba = ovr.predict_proba(X_test)
#Metriche
print("ACCURACY= "+str(accuracy_score(Y_test, Y_pred1)))
print("LOSS= "+str(log_loss(Y_test, Y_pred1_proba)))
cm1 = confusion_matrix(Y_test, Y_pred1)
#Matrice di confusione
plt.figure(figsize=(9,9))
sns.heatmap(cm1, annot=True, cmap="Blues_r", linewidths=.5, square=True)
plt.ylabel("Classe corrente")
plt.xlabel("Classe predetta")





