'''
Modello di classificazione di tipo K-nearest neighbor.
Test su dataset di numeri scritti a mano da classificare con il numero corretto corrispondente.
Dati: numeri scritti a mano
Target: numero vero associato
'''

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, accuracy_score
from sklearn.preprocessing import MinMaxScaler


digits = load_digits()
X = digits.data
Y = digits.target

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.3, random_state=0)

ms = MinMaxScaler()
X_train = ms.fit_transform(X_train)
X_test = ms.transform(X_test)

#Passiamo alla costruzione e predizione con modello K-NN
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn.fit(X_train, Y_train)
Y_pred = knn.predict(X_test)
Y_pred_proba = knn.predict_proba(X_test)

print("Test set metrics:")
print("ACC= "+str(accuracy_score(Y_test, Y_pred)))
print("LOGLOSS= "+str(log_loss(Y_test, Y_pred_proba)))

#Verifichiamo overfitting modello
Y_pred_train = knn.predict(X_train)
Y_pred_train_proba = knn.predict_proba(X_train)
print("(Overfitting check) Train set metrics:")
print("ACC= "+str(accuracy_score(Y_train, Y_pred_train)))
print("LOGLOSS= "+str(log_loss(Y_train, Y_pred_train_proba)))

print("*********************************************************")
'''
#Proviamo a far variare il valore di K
Ks = [1,2,3,4,5,7,10,12,15,20]
#Uso ciclo for per creare un modello per ciascun valore nella lista
for K in Ks:
    print("K="+str(K))
    knn = KNeighborsClassifier(n_neighbors=K)
    knn.fit(X_train, Y_train)
    
    Y_pred = knn.predict(X_test)
    Y_pred_proba = knn.predict_proba(X_test)

    print("Test set metrics:")
    print("ACC= "+str(accuracy_score(Y_test, Y_pred)))
    print("LOGLOSS= "+str(log_loss(Y_test, Y_pred_proba)))
    
    Y_pred_train = knn.predict(X_train)
    Y_pred_train_proba = knn.predict_proba(X_train)
    
    print("(Overfitting check) Train set metrics:")
    print("ACC= "+str(accuracy_score(Y_train, Y_pred_train)))
    print("LOGLOSS= "+str(log_loss(Y_train, Y_pred_train_proba)))
'''   
#==> K=3 e K=10, valori migliori del modello secondo le metriche

#Visualizziamo le immagini che il K=3 classifica erroneamente
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, Y_train)
Y_pred = knn.predict(X_test)

#Uso un ciclo che scandisca le previsioni del modello e in caso di errore stampa il numero corretto
for i in range(0,len(X_test)):
    if(Y_pred[i]!=Y_test[i]):
        print("Numero %d classificato come %d" %(Y_test[i], Y_pred[i]))
        #Per stampare il numero uso il metodo imshow che prende in input matrice di pixel
        plt.imshow(X_test[i].reshape([8,8]), cmap="gray")
        plt.show()




    
