import pandas as pd
import seaborn as sns
import sklearn
import numpy as np
import matplotlib.pyplot as plt




"""
Using the ccdefault dataset, with 90% for training and 10% for test (stratified sampling) 
and the decision tree model that you did in Module 2:
    
Part 1: Random test train splits
Run in-sample and out-of-sample accuracy scores for 10 different samples by changing random_state from 1 to 10 in sequence. 
Display the individual scores, then calculate the mean and standard deviation on the set of scores.  Report in a table format.
Part 2: Cross validation
Now rerun your model using cross_val_scores with k-fold CV (k=10).  
Report the individual fold accuracy scores, the mean CV score and the standard deviation of the fold scores. 
Now run the out-of-sample accuracy score.  Report in a table format.
Part 3: Conclusions
Write a short paragraph summarizing your findings.  
Which method of measuring accuracy provides the best estimate of how a model will do against unseen data?  
Which one is more efficient to run?
Part 4: Appendix
Link to github repo
"""

df=pd.read_csv("HW6_ccdefault.csv")

print(df.head())
print("\n")

from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

pipe=make_pipeline(StandardScaler(),DecisionTreeClassifier(criterion="gini",max_depth=2))

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
print("manual split")
accuracy_manual_out=list()
accuracy_manual_in=list()
for seed in range(1,11):
    X_train,X_test,y_train,y_test=train_test_split(df.iloc[:,1:24],df["DEFAULT"],test_size=0.1,random_state=seed,stratify=df["DEFAULT"])
    
    pipe.fit(X_train,y_train)
    
    y_pred=pipe.predict(X_test)
    y_pred_train=pipe.predict(X_train)
    
    accuracy_manual_in.append(accuracy_score(y_train,y_pred_train))
    accuracy_manual_out.append(accuracy_score(y_test,y_pred))

print(accuracy_manual_in)
print("\n")
print("the mean of the in-accuracy is",np.mean(accuracy_manual_in))
print("the standard deviation of the in-accuracy is",np.std(accuracy_manual_in,ddof=1))
print("\n")

print(accuracy_manual_out)
print("\n")
print("the mean of the out-accuracy is",np.mean(accuracy_manual_out))
print("the standard deviation of the out-accuracy is",np.std(accuracy_manual_out,ddof=1))
print("\n")


   
from sklearn.model_selection import cross_val_score
X_train,X_test,y_train,y_test=train_test_split(df.iloc[:,1:24],df["DEFAULT"],test_size=0.1,random_state=1,stratify=df["DEFAULT"])


pipe.fit(X_train,y_train)
print("cross validation")
accuracy_cross=cross_val_score(estimator=pipe,X=X_train,y=y_train,cv=10,n_jobs=1)
print(accuracy_cross)
print("\n")
print("the mean of the in-accuracy is",np.mean(accuracy_cross))
print("the standard deviation of the in-accuracy is",np.std(accuracy_cross,ddof=1))
print("\n")

y_pred_cross=pipe.predict(X_test)

print("the out-accuracy is",accuracy_score(y_test,y_pred_cross))


print("My name is Yi Zhou")
print("My NetID is: yizhou16")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")