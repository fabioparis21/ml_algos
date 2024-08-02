'''
Modello di albero decisionale
Database di passeggeri del Titanic con varie features e indica se sia sopravvisuto o meno
Target: il passeggero X sarebbe sopravvissuto all'incidente?
'''
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, accuracy_score

titanic = pd.read_csv("http://web.stanford.edu/class/archive/cs/cs109/cs109.1166/stuff/titanic.csv")
#==> 7 feature e un target, 887 esempi
#Elimino colonna con il nome
titanic = titanic.drop("Name", axis=1)

#One hot encoding per creare var. di comodo per la feature sesso
titanic = pd.get_dummies(titanic)

X = titanic.drop("Survived", axis=1).values
Y = titanic["Survived"].values
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.3, random_state=0)

#Usando un albero decisionale abbiamo il vantaggio di non dover portare tutti i dati sulla stessa scala!
from sklearn.tree import DecisionTreeClassifier
#Istanziamo la classe e come criterion inseriamo la metrica per le impurità
tree = DecisionTreeClassifier(criterion="gini", max_depth=6)
tree.fit(X_train, Y_train)

Y_pred = tree.predict(X_test)
Y_pred_proba = tree.predict_proba(X_test)
#Verifica overfitting
Y_pred_train = tree.predict(X_train)
Y_pred_train_proba = tree.predict_proba(X_train)

#Calcolo e stampa metriche
ll_pred = log_loss(Y_test, Y_pred_proba)
acc_pred = accuracy_score(Y_test, Y_pred)
ll_train = log_loss(Y_train, Y_pred_train_proba)
acc_train = accuracy_score(Y_train, Y_pred_train)

print("Metrics for test-set: ACC= %f , LOSS= %f " %(acc_pred,ll_pred))
print("Metrics for train-set: ACC= %f , LOSS= %f " %(acc_train,ll_train))

'''
==> Accuracy rispettive 78% pred e 98% trainset, modello troppo complicato in overfitting!
Cerchiamo di risolvere riducendo la profondità massima dell'albero, max_depth
Con max_depth=6 ottengo ==> Metrics for train-set: ACC= 0.893548 , LOSS= 0.278435 (much better!)

Altro vantaggio degli alberi è quello di poterli facilmente visualizzare:
'''
from sklearn.tree import export_graphviz
#Creiamo un file .dot
dotfile = open("tree.dot", "w")
export_graphviz(tree, out_file = dotfile, feature_names = titanic.columns.drop("Survived"))
dotfile.close()
