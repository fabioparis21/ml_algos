'''
Classificazione non lineare con kernel SVM
'''

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from viz import plot_bounds

#Creo un dataset con funzione di sklearn 
from sklearn.datasets import make_circles
#Genera un dataset di cerchi con rumore nei dati (disposti in modo non lineare)
X, Y = make_circles(noise=0.2, factor=0.5, random_state=1)   

#Creo scatterplot per vedere come sono distribuiti i dati
plt.scatter(X[:, 0], X[:, 1], c=Y)

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.3, random_state=0)

from sklearn.svm import SVC
#Istanzio la classe e scelgo il tipo di funzione kernel da usare
svc = SVC(kernel="linear", probability=True)  #Kernel linearee equivale ad una SVM classica
svc.fit(X_train, Y_train)

print("ACCURACY: Train= %.4f  Test= %.4f " % (svc.score(X_train, Y_train), svc.score(X_test, Y_test)) )
plot_bounds((X_train, X_test),(Y_train, Y_test), svc)

#Testiamo il modello con altri kernel
svc = SVC(kernel="rbf", probability=True)  #Kernel gaussiano, più utilizzato (parametro rbf)
svc.fit(X_train, Y_train)

print("ACCURACY: Train= %.4f  Test= %.4f " % (svc.score(X_train, Y_train), svc.score(X_test, Y_test)) )
plot_bounds((X_train, X_test),(Y_train, Y_test), svc)

svc = SVC(kernel="sigmoid", probability=True)  #Kernel sigmoidale
svc.fit(X_train, Y_train)

print("ACCURACY: Train= %.4f  Test= %.4f " % (svc.score(X_train, Y_train), svc.score(X_test, Y_test)) )
plot_bounds((X_train, X_test),(Y_train, Y_test), svc)

svc = SVC(kernel="poly", probability=True)  #Kernel polinomiale
svc.fit(X_train, Y_train)

print("ACCURACY: Train= %.4f  Test= %.4f " % (svc.score(X_train, Y_train), svc.score(X_test, Y_test)) )
plot_bounds((X_train, X_test),(Y_train, Y_test), svc)

