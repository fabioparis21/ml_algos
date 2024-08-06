'''
Non-linear classification with SVM Kernel algorithm
'''

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
#Custom function to plot the decision bounds of the model
from viz import plot_bounds

#Dataset creation
from sklearn.datasets import make_circles
#Generates a dataset of circles with noise, non-linear
X, Y = make_circles(noise=0.2, factor=0.5, random_state=1)   

#Plotting the dataset to see data distribution
plt.scatter(X[:, 0], X[:, 1], c=Y)

#Data preparation
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.3, random_state=0)

#Model creation
from sklearn.svm import SVC
svc = SVC(kernel="linear", probability=True)  #Linear kernel refers to classic SVM classification method
svc.fit(X_train, Y_train)

#Scoring
print("ACCURACY: Train= %.4f  Test= %.4f " % (svc.score(X_train, Y_train), svc.score(X_test, Y_test)) )
#Plotting decision bounds
plot_bounds((X_train, X_test),(Y_train, Y_test), svc)

#Model testing with different kernel types
svc = SVC(kernel="rbf", probability=True)  #Gaussian kernel
svc.fit(X_train, Y_train)

print("ACCURACY: Train= %.4f  Test= %.4f " % (svc.score(X_train, Y_train), svc.score(X_test, Y_test)) )
plot_bounds((X_train, X_test),(Y_train, Y_test), svc)

svc = SVC(kernel="sigmoid", probability=True)  #Sigmoidal kernel
svc.fit(X_train, Y_train)

print("ACCURACY: Train= %.4f  Test= %.4f " % (svc.score(X_train, Y_train), svc.score(X_test, Y_test)) )
plot_bounds((X_train, X_test),(Y_train, Y_test), svc)

svc = SVC(kernel="poly", probability=True)  #Poly kernel
svc.fit(X_train, Y_train)

print("ACCURACY: Train= %.4f  Test= %.4f " % (svc.score(X_train, Y_train), svc.score(X_test, Y_test)) )
plot_bounds((X_train, X_test),(Y_train, Y_test), svc)

