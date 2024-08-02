'''
Rete neurale - Percettrone multistrato
Dataset di numeri scritti a mano da classificare (60k esempi train, 10k esempi test)
'''

import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, accuracy_score
from sklearn.preprocessing import MinMaxScaler

#Funzione per estrarre direttamente il dataset
from mnist import load_mnist
X_train, X_test, Y_train, Y_test = load_mnist(path=r"C:\Users\paris\Desktop\script\Python\ML_lv1\mnist")

#L'input di una ANN deve essere sempre compreso tra 0 e 1
mms = MinMaxScaler()
X_train = mms.fit_transform(X_train)
X_test = mms.transform(X_test)

''' Essendo le ANN molto complesse con necessità di tuning per via dei numerosi parametri,
solitamente si utilizzanon solo quando strettamente necessario, quindi quando altri tipi di modelli
non ci danno informazioni adeguate al nostro obbiettivo'''

#Test con regressione logistica
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()
lr.fit(X_train, Y_train)

Y_pred = lr.predict(X_test)
Y_pred_proba = lr.predict_proba(X_test)
Y_pred_train = lr.predict(X_train)
Y_pred_train_proba = lr.predict_proba(X_train)

print("ACCURACY: Train= %.4f  Test=%.4f" % (accuracy_score(Y_train, Y_pred_train), accuracy_score(Y_test, Y_pred)))
print("LOGLOSS: Train= %.4f  Test=%.4f" % (log_loss(Y_train, Y_pred_train_proba), log_loss(Y_test, Y_pred_proba)))

#Proviamo ora con rete neurale (vanilla: 1 solo layer)
from sklearn.neural_network import MLPClassifier  #MultiLayerPercettron
#Istanzio e sceglo numero di hidden layers e num. di nodi per ciascun layer per la rete
mlp = MLPClassifier(hidden_layer_sizes=(100,), verbose=True)   #Un solo layer con 100 nodi
#"Verbose" ci permette di vedere in modo dinamico l'addestramento della rete
mlp.fit(X_train, Y_train)

Y_pred = mlp.predict(X_test)
Y_pred_proba = mlp.predict_proba(X_test)
Y_pred_train = mlp.predict(X_train)
Y_pred_train_proba = mlp.predict_proba(X_train)

print("ACCURACY: Train= %.4f  Test=%.4f" % (accuracy_score(Y_train, Y_pred_train), accuracy_score(Y_test, Y_pred)))
print("LOGLOSS: Train= %.4f  Test=%.4f" % (log_loss(Y_train, Y_pred_train_proba), log_loss(Y_test, Y_pred_proba)))

#Proviamo ora a rendere la rete "profonda" ossia con più hidden layers (deep)
mlp = MLPClassifier(hidden_layer_sizes=(512, 512), verbose=True)   #Due layers con 512 nodi ciascuno

mlp.fit(X_train, Y_train)

Y_pred = mlp.predict(X_test)
Y_pred_proba = mlp.predict_proba(X_test)
Y_pred_train = mlp.predict(X_train)
Y_pred_train_proba = mlp.predict_proba(X_train)

print("ACCURACY: Train= %.4f  Test=%.4f" % (accuracy_score(Y_train, Y_pred_train), accuracy_score(Y_test, Y_pred)))
print("LOGLOSS: Train= %.4f  Test=%.4f" % (log_loss(Y_train, Y_pred_train_proba), log_loss(Y_test, Y_pred_proba)))

#Vediamo quanti e quali esempi la rete ha fallito a classificare
err_count = 0
for i in range(len(X_test)):
    if (Y_test[i] != Y_pred[i]):
        err_count +=1
        print("Numero "+str(Y_test[i])+" classificato come "+str(Y_pred[i]))
        plt.imshow(X_test[i].reshape([28,28]), cmap="gray")
        plt.show()

err_perc = (err_count/len(X_test))*100

print("Numero errori totali: %d " %(err_count))
print("Percentuale d'errore su test set: %.4f " %(err_perc))
    