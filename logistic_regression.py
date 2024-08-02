'''
Modello di classificazione tramite regressione logistica su dataset di tumori al seno
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

breast_cancer = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data",
                           names=["id","diagnosis","radius_mean","texture_mean","perimeter_mean","area_mean","smoothness_mean",
                                  "compactness_mean","concavity_mean","concave points_mean","symmetry_mean","fractal_dimension_mean",
                                  "radius_se","texture_se","perimeter_se","area_se","smoothness_se","compactness_se","concavity_se",
                                  "concave points_se","symmetry_se","fractal_dimension_se","radius_worst","texture_worst",
                                  "perimeter_worst","area_worst","smoothness_worst","compactness_worst","concavity_worst",
                                  "concave points_worst","symmetry_worst","fractal_dimension_worst"])

#Osservo il dataset per vedere n.ro di esempi e il target
breast_cancer.info()
breast_cancer["diagnosis"].unique()  #==> Ci sono solo due classi M e B (maligno e benigno)

#Utilizziamo solo due features per iniziare a creare il modello
#Creazione arrays
X = breast_cancer[["radius_se", "concave points_worst"]].values
Y = breast_cancer["diagnosis"].values

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.3, random_state=0)

#Le classi target sono codificate come caratteri, la regr. logistica di scikit permette di usare anche queste.
#Utilizzo comunque LabelEncoder per ricodificarle come numeri.
le = LabelEncoder()
Y_train = le.fit_transform(Y_train)
Y_test = le.transform(Y_test)

#Ora standardizziamo il dataset con StandardScaler
ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)

#Ora eseguo regressione logistica
#Importo, istanzio la classe e addestro il modello.
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(X_train, Y_train)

#Passiamo al testing del modello con metriche di accuracy (% di predizioni corrette) e log-likelihood 
#(usiamo la log_loss ossia likelihood negativa, quindi valori minori equivalgono a modello migliore)
from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss
#Eseguo predizione su test set
Y_pred = lr.predict(X_test)
#Calcolo metriche
Y_pred_proba = lr.predict_proba(X_test)

print("ACCURACY= "+str(accuracy_score(Y_test, Y_pred)))  #Accuracy richiede in input le predizioni corrette (target test corretto)
print("LOG-LOSS= "+str(log_loss(Y_test, Y_pred_proba)))  #Log_loss richiede in input la probabilita di pred. corrette!

#Passo ora all'analisi del decision boundary del modello (confine oltre il quale un esempio verrà classificato come appartenente all'altra classe)
#utilizzo matplotlib per visualizzarlo graficamente tramite apposita funzione
def showBounds(model, X, Y, labels=["Negativo","Positivo"]):
    
    h = .02 

    x_min, x_max = X[:, 0].min(), X[:, 0].max()
    y_min, y_max = X[:, 1].min(), X[:, 1].max()

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])

    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.Paired)

    X_m = X[Y==1]
    X_b = X[Y==0]
    plt.scatter(X_b[:, 0], X_b[:, 1], c="green",  edgecolor='white', label=labels[0])
    plt.scatter(X_m[:, 0], X_m[:, 1], c="red",  edgecolor='white', label=labels[1])
    plt.legend()

#Plot su train set
showBounds(lr, X_train, Y_train, labels=["Benigno","Maligno"])
#Plot su test set
showBounds(lr, X_test, Y_test, labels=["Benigno","Maligno"])

#Proviamo ora a rieseguire il training del modello includendo tutte le features disponibili nel database
#tranne ID che sarebbe solo un identificativo dell'immagine non da nessuna info..
#Creo nuovi arrays
X1 = breast_cancer.drop(["diagnosis","id"], axis=1).values
Y1 = breast_cancer["diagnosis"].values

X1_train, X1_test, Y1_train, Y1_test = train_test_split(X1,Y1, test_size=0.3, random_state=0)

Y1_train = le.fit_transform(Y1_train)
Y1_test = le.transform(Y1_test)

X1_train = ss.fit_transform(X1_train)
X1_test = ss.transform(X1_test)

lr1 = LogisticRegression()
lr1.fit(X1_train, Y1_train)

Y1_pred = lr1.predict(X1_test)
Y1_pred_proba = lr1.predict_proba(X1_test)

print("Metrics with all features:")
print("ACCURACY= "+str(accuracy_score(Y1_test, Y1_pred)))  
print("LOG-LOSS= "+str(log_loss(Y1_test, Y1_pred_proba)))
#==> Indicatori ci dicono che il modello è effettivamente più preciso con tutte le features!

#NB: La reg. logistica implementa la regolarizzazione del modello con 2 parametri: penalty(l2 o l1) e C(inverso di lambda)
#quindi un valore maggiore di C rende la regolarizzazione più debole, di default abbiamo (penalty="l2", C=1).

