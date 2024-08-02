#Modello di regressione polinomiale

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


boston = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data",
                     sep="\s+", names=["CRIM","ZN","INDUS","CHAS","NOX","RM","AGE","DIS","RAD","TAX","PRATIO","B","LSTAT","MEDV"] )

#Creo lista con proprietà con maggiore incidenza sul prezzo
cols = ["RM", "LSTAT", "DIS", "RAD", "MEDV"]
#Creo plot sulle proprietà di interesse per osservarne le relazioni
sns.pairplot(boston[cols])   #==>Relazione tra LSTAT e MEDV è non lineare! Approssimabile solo con una curva

#Creazione arrays
X = boston[["LSTAT"]].values
Y = boston[["MEDV"]].values
#Creazione set
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)

#Per eseguire una regr. polinomiale dobbiamo creare delle proprietà aggiuntive che altro non sono che la combinazione
#polinomiale delle prop. che già abbiamo, fino al grado desiderato.
#Usiamo classe polynomial features da istanziare con grado del poli. che vogliamo
from sklearn.preprocessing import PolynomialFeatures

#Vogliamo confrontare i risultati ottenibili con i diversi gradi del polinomio
#Ciclo che crea di volta in volta una regressione partendo dal grado 1 al grado 11 e ne stampa indici di bontà
for i in range(1,11):
    polyft = PolynomialFeatures(degree=i)
    X_train_poly = polyft.fit_transform(X_train)
    X_test_poly = polyft.transform(X_test)
    
    ll = LinearRegression()
    ll.fit(X_train_poly, Y_train)
    Y_pred = ll.predict(X_test_poly)
    
    mse = mean_squared_error(Y_test, Y_pred)
    r2 = r2_score(Y_test, Y_pred)    
    
    print("DEGREE:"+str(i)+"  Mse="+str(mse)+"  R2="+str(r2))
    
    





