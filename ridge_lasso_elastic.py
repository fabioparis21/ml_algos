'''
Partendo dal modello in overfitting, applico la regolarizzazione.
Fase di data processing con pandas per lettura dataset e creazione dataframe.
Preparazione dati per modello con creazione array features(X) e array target(Y), aggiunta features
polinomiali e standardizzazione.
Modelli lineari regolarizzati Ridge, Lasso ed ElasticNet con ciclo che ne valuta gli indici
mean squared error e r2 ratio.
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

boston = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data",
                     sep="\s+", names=["CRIM","ZN","INDUS","CHAS","NOX","RM","AGE","DIS","RAD","TAX","PRATIO","B","LSTAT","MEDV"] )

X = boston.drop("MEDV", axis=1).values
Y = boston["MEDV"].values

#Split arrays
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.3, random_state=0)
#Aggiunta features polinomiali agli array (preprocessing)
polyft = PolynomialFeatures(degree=2)
X_train_poly = polyft.fit_transform(X_train)
X_test_poly = polyft.transform(X_test)

#Standardizzazione array (preprocessing)
ss = StandardScaler()
X_train_poly = ss.fit_transform(X_train_poly)
X_test_poly = ss.transform(X_test_poly)

#Creazione modello 
ll = LinearRegression()
ll.fit(X_train_poly, Y_train)
#Predizione del modello usando i dati di addestramento (OVERFITTING)
Y_pred_train = ll.predict(X_train_poly)
#I risultati di affidabilità del modello troppo elevata sono un campanello d'allarme
mse_train = mean_squared_error(Y_train, Y_pred_train)
r2_train = r2_score(Y_train, Y_pred_train)
print("MSE= "+str(mse_train)+" R2= "+str(r2_train))  #==>Indice di errore <5 e affidabilità r2 >95 !

#Predizione del modello usando i dati di test
Y_pred_test = ll.predict(X_test_poly)
#I risultati di affidabilità del modello ora sono di molto differenti, peggiori
mse_test = mean_squared_error(Y_test, Y_pred_test)
r2_test = r2_score(Y_test, Y_pred_test)
print("MSE= "+str(mse_test)+" R2= "+str(r2_test))  #==>Indice di errore >25 e affidabilità r2 <65 !

#Utilizziamo regolarizzazione di tipo L2 con Ridge
from sklearn.linear_model import Ridge
#Creiamo un array dei possibili valori del parametro (qui alpha, chiamato anche lambda)
alphas = [0.0001, 0.001, 0.01, 0.1, 1., 10.]
#Creiamo un ciclo che crei diversi modelli per i vari valori di alpha scelti
print("Ridge cycle:")
for alpha in alphas:
    #Stampa alpha attuale e lo imposta al modello
    print("ALPHA= "+str(alpha))
    model = Ridge(alpha=alpha)
    model.fit(X_train_poly, Y_train)
    
    #Predizione su entrambe i set come prima
    Y_pred_train = model.predict(X_train_poly)
    Y_pred_test = model.predict(X_test_poly)
    #Stampa degli indicatori
    mse_train = mean_squared_error(Y_train, Y_pred_train)
    r2_train = r2_score(Y_train, Y_pred_train)
    print("Train set: "+"MSE= "+str(mse_train)+" R2= "+str(r2_train))
    
    mse_test = mean_squared_error(Y_test, Y_pred_test)
    r2_test = r2_score(Y_test, Y_pred_test)
    print("Test set: "+"MSE= "+str(mse_test)+" R2= "+str(r2_test))

#Utilizziamo regolarizzazione L1 con Lasso
from sklearn.linear_model import Lasso
print("Lasso cycle:")
for alpha in alphas:
    #Stampa alpha attuale e lo imposta al modello
    print("ALPHA= "+str(alpha))
    model = Lasso(alpha=alpha)
    model.fit(X_train_poly, Y_train)
    
    #Predizione su entrambe i set come prima
    Y_pred_train = model.predict(X_train_poly)
    Y_pred_test = model.predict(X_test_poly)
    #Stampa degli indicatori
    mse_train = mean_squared_error(Y_train, Y_pred_train)
    r2_train = r2_score(Y_train, Y_pred_train)
    print("Train set: "+"MSE= "+str(mse_train)+" R2= "+str(r2_train))
    
    mse_test = mean_squared_error(Y_test, Y_pred_test)
    r2_test = r2_score(Y_test, Y_pred_test)
    print("Test set: "+"MSE= "+str(mse_test)+" R2= "+str(r2_test))    
    
#Ora utilizzo modello ElasticNet che combina le due tecniche di reg. per un risultato migliore
from sklearn.linear_model import ElasticNet
#Riutilizziamo lo stesso codice per avere un confronto fra le tre tecniche
print("ElastciNet cycle:")
for alpha in alphas:
    #Stampa alpha attuale e lo imposta al modello
    print("ALPHA= "+str(alpha))
    #Questo modello ha un secondo parametro l1_ratio che ci permette di impostare a quale tipo di
    #reg. dare più importanza
    model = ElasticNet(alpha=alpha, l1_ratio=0.5)
    model.fit(X_train_poly, Y_train)
    
    #Predizione su entrambe i set come prima
    Y_pred_train = model.predict(X_train_poly)
    Y_pred_test = model.predict(X_test_poly)
    #Stampa degli indicatori
    mse_train = mean_squared_error(Y_train, Y_pred_train)
    r2_train = r2_score(Y_train, Y_pred_train)
    print("Train set: "+"MSE= "+str(mse_train)+" R2= "+str(r2_train))
    
    mse_test = mean_squared_error(Y_test, Y_pred_test)
    r2_test = r2_score(Y_test, Y_pred_test)
    print("Test set: "+"MSE= "+str(mse_test)+" R2= "+str(r2_test))  
#==>Risultati di molto migliori rispetto al singolo uso di L1/L2
#(NB) Prima di applicare la regolarizzazione assicurarsi che tutti i dati siano sulla stessa scala!!
#(standardizzati o normalizzati)
    



