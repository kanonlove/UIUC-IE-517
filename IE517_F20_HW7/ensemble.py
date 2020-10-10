import pandas as pd
import seaborn as sns
import sklearn
import numpy as np
import matplotlib.pyplot as plt
import time

df=pd.read_csv("HW7_ccdefault.csv")

print(df.head())
print("\n")

"""
Using the ccdefault dataset, and 10 fold cross validation described in Raschka; 
Part 1: Random forest estimators
Fit a random forest model, try several different values for N_estimators, report in-sample accuracies. 
Part 2: Random forest feature importance
Display the individual feature importance of your best model in Part 1 above using the code presented in Chapter 4 on page 136. 
{importances=forest.feature_importances_ }
Part 3: Conclusions
Write a short paragraph summarizing your findings. Answer the following questions:
a)	What is the relationship between n_estimators, in-sample CV accuracy and computation time?
b)	What is the optimal number of estimators for your forest?  
c)	Which features contribute the most importance in your model according to scikit-learn function?  
d)	What is feature importance and how is it calculated?  (If you are not sure, refer to the Scikit-Learn.org documentation.)
Part 4: Appendix
Link to github repo

"""


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X_train,X_test,y_train,y_test=train_test_split(df.iloc[:,1:24],df["DEFAULT"],test_size=0.1,random_state=1,stratify=df["DEFAULT"])
scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)



from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
feature_labels=df.columns[1:24]

n_estimators= [10,25,50,100,250,500,750,1000]
for number in n_estimators:
    #start time
    start_time=time.time()
    forest=RandomForestClassifier(n_estimators=number,n_jobs=-1)
    forest.fit(X_train,y_train)
    accuracy_cross=cross_val_score(estimator=forest,X=X_train,y=y_train,cv=10,n_jobs=-1)
    #end time
    end_time=time.time()
    print(number, "estimator accuracy is",np.mean(accuracy_cross))
    print("training time is",end_time-start_time,"seconds")

# though 500 is the best in accuracy, I would suggest using 25 or 50
forest=RandomForestClassifier(n_estimators=500,n_jobs=-1)
forest.fit(X_train,y_train)
importances = pd.Series(data=forest.feature_importances_,index=feature_labels)
importances_sorted = importances.sort_values(ascending=False)
    
plt.figure(figsize=(15,5))
importances_sorted.plot(kind="bar")
plt.title('Features Importances')
plt.show()
    
print(importances)
print("\n")


print("My name is Yi Zhou")
print("My NetID is: yizhou16")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")