'''
Multiclass classification with one vs all technique that uses multiple binary classifications.
Hand written numbers dataset from sklearn
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, accuracy_score
from sklearn.datasets import load_digits

digits = load_digits()
X = digits.data
Y = digits.target
#==>Dataset has 1797 examples with 64 features, every feature is a pixel.

#Cycle to visualise the pixel matrix
for i in range(0,10):
    pic_matrix = X[Y==i][0].reshape([8,8])
    plt.imshow(pic_matrix, cmap="gray")
    plt.show()

#Data preparation phase
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.3, random_state=0)

from sklearn.preprocessing import MinMaxScaler
mms = MinMaxScaler()
X_train = mms.fit_transform(X_train)
X_test = mms.transform(X_test)

#Modeling phase
#LogisticRegression method automatically recognizes the multiclass method and applies OneVSAll technique
lr = LogisticRegression()
lr.fit(X_train, Y_train)
Y_pred = lr.predict(X_test)
Y_pred_proba = lr.predict_proba(X_test)

#Scoring phase
print("ACCURACY= "+str(accuracy_score(Y_test, Y_pred)))
print("LOSS= "+str(log_loss(Y_test, Y_pred_proba)))

#We use the confusion matrix to see which classes the model failed to classify the most 
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, Y_pred)

#Passiamo alla visualizzazione della matrice generata
import seaborn as sns
'''
We use an heatmap that visualize for each class which and how much errors the model has done
'''
plt.figure(figsize=(9,9))
sns.heatmap(cm, annot=True, cmap="Blues_r", linewidths=.5, square=True)
plt.ylabel("Current class")
plt.xlabel("Predicted class")

#Sklearn also has another type of OneVSAll classifier
from sklearn.multiclass import OneVsRestClassifier
#The method requires a basic classifier to use
ovr = OneVsRestClassifier(LogisticRegression())

ovr.fit(X_train,Y_train)
Y_pred1 = ovr.predict(X_test)
Y_pred1_proba = ovr.predict_proba(X_test)
#Metrics
print("ACCURACY= "+str(accuracy_score(Y_test, Y_pred1)))
print("LOSS= "+str(log_loss(Y_test, Y_pred1_proba)))
cm1 = confusion_matrix(Y_test, Y_pred1)
#Confusion matrix
plt.figure(figsize=(9,9))
sns.heatmap(cm1, annot=True, cmap="Blues_r", linewidths=.5, square=True)
plt.ylabel("Current class")
plt.xlabel("Predicted class")





