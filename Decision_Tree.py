'''
Decision tree model
Using Titanic passengers database.
Target: did the passenger survive the accident?
'''
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, accuracy_score

titanic = pd.read_csv("http://web.stanford.edu/class/archive/cs/cs109/cs109.1166/stuff/titanic.csv")
#==> 7 features, 1 target, 887 examples

#Data preparation phase
titanic = titanic.drop("Name", axis=1)

#One hot encoding on the sex feature
titanic = pd.get_dummies(titanic)

X = titanic.drop("Survived", axis=1).values
Y = titanic["Survived"].values
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.3, random_state=0)

#Modeling phase
#Using this model type, we do not need to bring all the entries on the same scale
from sklearn.tree import DecisionTreeClassifier

tree = DecisionTreeClassifier(criterion="gini", max_depth=6)
tree.fit(X_train, Y_train)

Y_pred = tree.predict(X_test)
Y_pred_proba = tree.predict_proba(X_test)
#Overfitting check
Y_pred_train = tree.predict(X_train)
Y_pred_train_proba = tree.predict_proba(X_train)

#Scoring phase
ll_pred = log_loss(Y_test, Y_pred_proba)
acc_pred = accuracy_score(Y_test, Y_pred)
ll_train = log_loss(Y_train, Y_pred_train_proba)
acc_train = accuracy_score(Y_train, Y_pred_train)

print("Metrics for test-set: ACC= %f , LOSS= %f " %(acc_pred,ll_pred))
print("Metrics for train-set: ACC= %f , LOSS= %f " %(acc_train,ll_train))

'''
==> Accuracy 78% prediction and 98% trainset, the model is too complex, overfitting!
The solution is reducing the complexity through the max_depth parameter
With max_depth=6 we obtain ==> Metrics for train-set: ACC= 0.893548 , LOSS= 0.278435 (much better!)

We can also easily visualize the decision tree exporting the dotfile:
'''

from sklearn.tree import export_graphviz
dotfile = open("tree.dot", "w")
export_graphviz(tree, out_file = dotfile, feature_names = titanic.columns.drop("Survived"))
dotfile.close()
