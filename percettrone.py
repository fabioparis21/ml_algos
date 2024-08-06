'''
Neural network with multi-layer perceptron
using Mnist hand-written number dataset (60k examples for train and 10k for test)
'''

import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, accuracy_score
from sklearn.preprocessing import MinMaxScaler

#Custom function to extract the dataset 
from mnist import load_mnist
X_train, X_test, Y_train, Y_test = load_mnist(path) #Insert here the path of the folder which contains the mnist dataset

#The input for an ANN must be a number between 0 and 1
mms = MinMaxScaler()
X_train = mms.fit_transform(X_train)
X_test = mms.transform(X_test)

#Test with logistic regression 
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()
lr.fit(X_train, Y_train)

Y_pred = lr.predict(X_test)
Y_pred_proba = lr.predict_proba(X_test)
Y_pred_train = lr.predict(X_train)
Y_pred_train_proba = lr.predict_proba(X_train)

print("ACCURACY: Train= %.4f  Test=%.4f" % (accuracy_score(Y_train, Y_pred_train), accuracy_score(Y_test, Y_pred)))
print("LOGLOSS: Train= %.4f  Test=%.4f" % (log_loss(Y_train, Y_pred_train_proba), log_loss(Y_test, Y_pred_proba)))

#Test with vanilla type network (1 layer)
from sklearn.neural_network import MLPClassifier  
#Modeling phase
mlp = MLPClassifier(hidden_layer_sizes=(100,), verbose=True)   #One layer with 100 nodes

#"Verbose" allow us to see real-time training results for each epoch
mlp.fit(X_train, Y_train)

#Scoring phase
Y_pred = mlp.predict(X_test)
Y_pred_proba = mlp.predict_proba(X_test)
Y_pred_train = mlp.predict(X_train)
Y_pred_train_proba = mlp.predict_proba(X_train)

print("ACCURACY: Train= %.4f  Test=%.4f" % (accuracy_score(Y_train, Y_pred_train), accuracy_score(Y_test, Y_pred)))
print("LOGLOSS: Train= %.4f  Test=%.4f" % (log_loss(Y_train, Y_pred_train_proba), log_loss(Y_test, Y_pred_proba)))

#Testing with a deep network
mlp = MLPClassifier(hidden_layer_sizes=(512, 512), verbose=True)   #2 layers with 512 nodes each

mlp.fit(X_train, Y_train)

#Scoring phase
Y_pred = mlp.predict(X_test)
Y_pred_proba = mlp.predict_proba(X_test)
Y_pred_train = mlp.predict(X_train)
Y_pred_train_proba = mlp.predict_proba(X_train)

print("ACCURACY: Train= %.4f  Test=%.4f" % (accuracy_score(Y_train, Y_pred_train), accuracy_score(Y_test, Y_pred)))
print("LOGLOSS: Train= %.4f  Test=%.4f" % (log_loss(Y_train, Y_pred_train_proba), log_loss(Y_test, Y_pred_proba)))

#This code allow us to see and plot which examples the network failed to classify
err_count = 0
for i in range(len(X_test)):
    if (Y_test[i] != Y_pred[i]):
        err_count +=1
        print("Number "+str(Y_test[i])+" classified as "+str(Y_pred[i]))
        plt.imshow(X_test[i].reshape([28,28]), cmap="gray")
        plt.show()

#Percentage of errors made by the model
err_perc = (err_count/len(X_test))*100

print("Total errors: %d " %(err_count))
print("Error percentage on test set: %.4f " %(err_perc))
    
