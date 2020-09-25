import pandas as pd
import seaborn as sns
import sklearn
import numpy as np
from pandas import DataFrame
import matplotlib.pyplot as plt
import time


"""
Part 1: Exploratory Data Analysis
Describe the data set sufficiently using the methods and visualizations that we used previously.  
Include any output, graphs, tables, that you think is necessary to represent the data. 
 Label your figures and axes. DO NOT INCLUDE CODE, only output figures!
Split data into training and test sets.  Use random_state = 42. 
Use 85% of the data for the training set.  Use the same split for all experiments.

Part 2: Perform a PCA on the Treasury Yield dataset
Compute and display the explained variance ratio for all components, then recalculate and display on n_components=3.
What is the cumulative explained variance of the 3 component version.

Part 3: Linear regression v. SVM regressor - baseline
Fit a linear regression model to both datasets (the original dataset with 30 attributes and the PCA transformed dataset with 3 PCs.)
 using SKlearn.  Calculate its accuracy R2 score and RMSE for both in sample and out of sample (train and test sets).  
 (You may use CV accuracy score if you wish).
Fit a SVM regressor model to both datasets using SKlearn.  
Calculate its accuracy R2 score and RMSE for both in sample and out of sample (train and test sets).  
(You may use CV accuracy score if you wish).

Part 4: Conclusions
Write a short paragraph summarizing your findings.  
Which model performs best on the untransformed data?  
Which transformation leads to the best performance increases?  
How does training time change for the two models.  
Report your results using the Results worksheet format.  
Embed the completed table in your report. 

Part 5: Appendix
Link to github repo

"""


df=pd.read_csv("hw5_treasury yield curve data.csv")
print(df.head())
print("\n")


# heat map
cormat = DataFrame(df.corr())

#visualize correlations using heatmap
fig, ax = plt.subplots(figsize=(20,20))
sns.heatmap(cormat,linewidth=1,square=True,ax=ax,annot=True)
ax.set_ylim([31, 0])
plt.xlabel("Heat Map of treasury yield curve")
plt.show()
print("\n")


# scatter plot
# obviously all the features are highly correlated
# here i pick four features 
# two of them are close, others are far away
# and i did the scatter plot to the target
cols=["SVENF01","SVENF02","SVENF15","SVENF30","Adj_Close"]
sns.pairplot(df[cols], height=2.5,diag_kws=dict(edgecolor="black",bins=5),plot_kws=dict(s=10,linewidth=0.2))
plt.tight_layout()
plt.show()

print("\n")

#data split
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
features_treasury=df.iloc[:,1:31]
Adj_Close=df.iloc[:,31]
X_train, X_test, y_train, y_test = train_test_split(features_treasury, Adj_Close,test_size=0.15, random_state=42)

scaler = preprocessing.StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)



#linear regression

#start time
start_linear_regression=time.time()


print("Linear Regression")
from sklearn.linear_model import LinearRegression

linear_reg=LinearRegression()
linear_reg.fit(X_train,y_train)

y_test_linear_predict=linear_reg.predict(X_test)
y_train_linear_predict=linear_reg.predict(X_train)

#end time
end_linear_regression=time.time()

print("linear regression training time before PCA:",end_linear_regression-start_linear_regression)
print("\n")

"""
#coefficient and intercept for each column
counter=0
for value in linear_reg.coef_:
    print(df.columns[counter+1],"coefficient:",value)
    counter=counter+1
print("\n")
print("intercept is",linear_reg.intercept_)
print("\n")
"""

#RMSE and R^2
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from math import sqrt

print("RMSE Train:")
print (sqrt(mean_squared_error(y_train_linear_predict,y_train)))

print("RMSE Test:")
print(sqrt(mean_squared_error(y_test_linear_predict,y_test)))
print("\n")
print("R^2 Train:")
print (r2_score(y_train_linear_predict,y_train))

print("R^2 Test:")
print (r2_score(y_test_linear_predict,y_test))
print("\n")


# SVM
print("Support Vector Regression")
from sklearn.svm import SVR

#start time
start_SVR=time.time()
SVM_reg=SVR(kernel="linear")
SVM_reg.fit(X_train,y_train)


y_test_SVM_predict=SVM_reg.predict(X_test)
y_train_SVM_predict=SVM_reg.predict(X_train)

#end time
end_SVR=time.time()


print("SVR training time before PCA:",end_SVR-start_SVR)
print("\n")

"""
#coefficient
counter=0
for value in SVM_reg.coef_[0]:
    print(df.columns[counter+1],"coefficient:",value)
    counter=counter+1
print("\n")

print("intercept is",SVM_reg.intercept_[0])
"""

#RMSE and R^2

print("RMSE Train:")
print (sqrt(mean_squared_error(y_train_SVM_predict,y_train)))

print("RMSE Test:")
print(sqrt(mean_squared_error(y_test_SVM_predict,y_test)))
print("\n")

print("R^2 Train:")
print (r2_score(y_train_SVM_predict,y_train))

print("R^2 Test:")
print (r2_score(y_test_SVM_predict,y_test))
print("\n")


#PCA 
from sklearn.decomposition import PCA

# explained variance plot befor PCA
cov_matrix=np.cov(X_train.T)
eigenvalues,eigenvectors=np.linalg.eig(cov_matrix)

#print(eigenvalues)
sorted(eigenvalues,reverse=True)

tot = sum(eigenvalues)

percentage_first_three=(eigenvalues[0]+eigenvalues[1]+eigenvalues[2])/tot
print("cumulative percentage of first three principal components:",percentage_first_three)

var_exp = [(i / tot) for i in sorted(eigenvalues, reverse=True)]
cum_var_exp = np.cumsum(var_exp)

plt.bar(range(1, 31), var_exp, alpha=0.5, align='center',label='individual explained variance')
plt.step(range(1, 31), cum_var_exp, where='mid',label='cumulative explained variance')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal component index')
plt.legend(loc='best')
plt.tight_layout()
plt.show()

#PCA transformation
pca=PCA(n_components=3)
X_train_pca=pca.fit_transform(X_train)
X_test_pca=pca.transform(X_test)

#explained variance plot after PCA
cov_matrix_pca=np.cov(X_train_pca.T)
eigenvalues_pca,eigenvectors_pca=np.linalg.eig(cov_matrix_pca)
#print(eigenvalues)

tot_pca = sum(eigenvalues_pca)
var_exp_pca = [(i / tot_pca) for i in sorted(eigenvalues_pca, reverse=True)]
cum_var_exp_pca = np.cumsum(var_exp_pca)

plt.bar(range(1, 4), var_exp_pca, alpha=0.5, align='center',label='individual explained variance')
plt.step(range(1, 4), cum_var_exp_pca, where='mid',label='cumulative explained variance')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal component index')
plt.legend(loc='best')
plt.tight_layout()
plt.show()







# linear regression after PCA
print("Linear Regression after PCA")
#start time
start_linear_regression_pca=time.time()

linear_reg_pca=LinearRegression()
linear_reg_pca.fit(X_train_pca,y_train)
y_test_linear_pca_predict=linear_reg_pca.predict(X_test_pca)
y_train_linear_pca_predict=linear_reg_pca.predict(X_train_pca)

#end time
end_linear_regression_pca=time.time()

print("linear regression training time after PCA:",end_linear_regression_pca-start_linear_regression_pca)
print("\n")

"""
#coefficient and intercept for each column
counter=0
for value in linear_reg_pca.coef_:
    print(df.columns[counter+1],"coefficient:",value)
    counter=counter+1
print("\n")
print("intercept is",linear_reg_pca.intercept_)
print("\n")
"""

#RMSE and R^2
print("RMSE Train:")
print(sqrt(mean_squared_error(y_train_linear_pca_predict,y_train)))

print("RMSE Test:")
print(sqrt(mean_squared_error(y_test_linear_pca_predict,y_test)))
print("\n")
print("R^2 Train:")
print (r2_score(y_train_linear_pca_predict,y_train))

print("R^2 Test:")
print (r2_score(y_test_linear_pca_predict,y_test))
print("\n")








#SVM after PCA
print("Support Vector Regression after PCA")
#start time
start_SVR_pca=time.time()

SVM_reg_pca=SVR(kernel="linear")
SVM_reg_pca.fit(X_train_pca,y_train)
y_test_SVM_pca_predict=SVM_reg_pca.predict(X_test_pca)
y_train_SVM_pca_predict=SVM_reg_pca.predict(X_train_pca)

#end time
end_SVR_pca=time.time()

print("SVR training time after PCA:", end_SVR_pca-start_SVR_pca)
print("\n")

"""
#coefficient
counter=0
for value in SVM_reg_pca.coef_[0]:
    print(df.columns[counter+1],"coefficient:",value)
    counter=counter+1
print("\n")

print("intercept is",SVM_reg_pca.intercept_[0])
"""

#RMSE and R^2

print("RMSE Train:")
print (sqrt(mean_squared_error(y_train_SVM_pca_predict,y_train)))

print("RMSE Test:")
print(sqrt(mean_squared_error(y_test_SVM_pca_predict,y_test)))
print("\n")

print("R^2 Train:")
print (r2_score(y_train_SVM_pca_predict,y_train))

print("R^2 Test:")
print (r2_score(y_test_SVM_pca_predict,y_test))
print("\n")

print("My name is Yi Zhou")
print("My NetID is: yizhou16")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")
