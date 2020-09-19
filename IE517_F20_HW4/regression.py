import pandas as pd
import seaborn as sns
import numpy as np
from pandas import DataFrame
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
import matplotlib.pyplot as plt;


"""
Part 1: Exploratory Data Analysis
Describe the data sufficiently using the methods and visualizations that we used 
previously in Module 3 and again this week.  Include any output, graphs, tables, heatmaps, 
box plots, etc.  Label your figures and axes. DO NOT INCLUDE CODE!
Split data into training and test sets.  Use random_state = 42. Use 80% of the data
 for the training set.  Use the same split for all models.


Part 2: Linear regression
Fit a linear model using SKlearn to all of the features of the dataset.  
Describe the model (coefficients and y intercept), plot the residual errors, 
calculate performance metrics: MSE and R2.  



Part 3.1: Ridge regression
Fit a Ridge model using SKlearn to all of the features of the dataset.  
Test some settings for alpha.  Describe the model (coefficients and y intercept), 
plot the residual errors, calculate performance metrics: MSE and R2.  Which alpha gives the best performing model?


Part 3.2: LASSO regression
Fit a LASSO model using SKlearn to all of the features of the dataset.  
Test some settings for alpha.  Describe the model (coefficients and y intercept), 
plot the residual errors, calculate performance metrics: MSE and R2.  Which alpha gives the best performing model?


Part 4: Conclusions
Write a short paragraph summarizing your findings.  


Part 5: Appendix
Link to github repo

"""

df=pd.read_csv("HW4_housing.csv")

print(df.head())

#print(len(df))

print("\n")

"""
• CRIM: Per capita crime rate by town
• ZN: Proportion of residential land zoned for lots over 25,000 sq. ft.
• INDUS: Proportion of non-retail business acres per town
• CHAS: Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
• NOX: Nitric oxide concentration (parts per 10 million)
• RM: Average number of rooms per dwelling
• AGE: Proportion of owner-occupied units built prior to 1940
• DIS: Weighted distances to five Boston employment centers
• RAD: Index of accessibility to radial highways
• TAX: Full-value property tax rate per $10,000
• PTRATIO: Pupil-teacher ratio by town
• B: 1000(Bk - 0.63)^2, where Bk is the proportion of [people of African American descent] by town
• LSTAT: Percentage of lower status of the population
• MEDV: Median value of owner-occupied homes in $1000s (TARGET)

"""
   

# heat map
cormat = DataFrame(df.corr())
#visualize correlations using heatmap
fig, ax = plt.subplots(figsize=(10,10))
sns.heatmap(cormat,linewidth=1,square=True,ax=ax,annot=True)
ax.set_ylim([14, 0])
plt.xlabel("Heat Map of Boston House Price")
plt.show()
print("\n")



#CRIM
plt.hist(df["CRIM"],edgecolor="black",bins=10)
plt.xlabel("per capita crime rate by town")
plt.ylabel("numbers")
plt.show()



#ZN
plt.hist(df["ZN"],edgecolor="black",bins=10)
plt.xlabel("ZProportion of residential land zoned for lots over 25,000 sq. ft")
plt.ylabel("numbers")
plt.show()



#INDUS
plt.hist(df["INDUS"],edgecolor="black",bins=[0,2.5,5,7.5,10,12.5,15,17.5,20,22.5,25,27.5,30])
plt.xlabel("Proportion of non-retail business acres per town")
plt.ylabel("numbers")
plt.show()




#CHAS
false_number=0
true_number=0
for value in df["CHAS"]:
    if value== 0:
        
        false_number=false_number+1
    else:
        
        true_number=true_number+1
    


plt.bar(["tract bounding river","tract not bounding river"],height=[true_number,false_number],width=0.4,edgecolor="black")
plt.xlabel("Charles River dummy variable")
plt.ylabel("Number")
plt.show()



#NOX
plt.hist(df["NOX"],edgecolor="black",bins=[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])
plt.xlabel("Nitric oxide concentration (parts per 10 million)")
plt.ylabel("numbers")
plt.show()

#RM
plt.hist(df["RM"],edgecolor="black",bins=[0,1,2,3,4,5,6,7,8,9,10])
plt.xlabel("Average number of rooms per dwelling")
plt.ylabel("numbers")
plt.show()



#AGE
plt.hist(df["AGE"],edgecolor="black",bins=[0,10,20,30,40,50,60,70,80,90,100,110])
plt.xlabel("Proportion of owner-occupied units built prior to 1940")
plt.ylabel("numbers")
plt.show()



#DIS
plt.hist(df["DIS"],edgecolor="black",bins=[0,2,4,6,8,10,12,14])
plt.xlabel("Weighted distances to five Boston employment centers")
plt.ylabel("numbers")
plt.show()



#RAD
#this one is more like binary data
#I am not sure what the "data" are
#I might also do a hist for the data below 10
below_10_number=0
index_24_number=0

for value in df["RAD"]:
    if value == 24:
        
        index_24_number=index_24_number+1
    
    
    else:
        
        below_10_number=below_10_number+1
    
plt.bar(["index below 10","index is 24"],height=[below_10_number,index_24_number],width=0.4,edgecolor="black")
plt.xlabel("Index of accessibility to radial highways")
plt.ylabel("Number")
plt.show()

plt.hist(df["RAD"],edgecolor="black",bins=[0,2,4,6,8,10,12,14,16,18,20,24,26])
plt.xlabel("Index of accessibility to radial highways")
plt.ylabel("numbers")
plt.show()



#TAX
plt.hist(df["TAX"],edgecolor="black",bins=[0,100,200,300,400,500,600,700,800])
plt.xlabel("Full-value property tax rate per $10,000")
plt.ylabel("numbers")
plt.show()


#PTRATIO 
plt.hist(df["PTRATIO"],edgecolor="black",bins=[0,2.5,5,7.5,10,12.5,15,17.5,20,22.5,25])
plt.xlabel("Pupil-teacher ratio by town")
plt.ylabel("numbers")
plt.show()


#B
plt.hist(df["B"],edgecolor="black",bins=[0,50,100,150,200,250,300,350,400])
plt.xlabel("1000(Bk - 0.63)^2,Bk is the proportion of people of African American descent by town")
plt.ylabel("numbers")
plt.show()


#LSTAT
plt.hist(df["LSTAT"],edgecolor="black",bins=[0,5,10,15,20,25,30,35,40])
plt.xlabel("Percentage of lower status of the population")
plt.ylabel("numbers")
plt.show()


#MEDV
plt.hist(df["MEDV"],edgecolor="black",bins=[0,5,10,15,20,25,30,35,40,45,50,55,60])
plt.xlabel("Median value of owner-occupied homes in $1000s (TARGET)")
plt.ylabel("numbers")
plt.show()


#exploratory ends here

print("\n")
print("Linear Regression without penalty")
print("\n")

#split the data
#standardize the data
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
X,y=df.iloc[:,0:13],df.iloc[:,13]



X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2, random_state=42)

#print( X_train.shape, y_train.shape)

scaler = preprocessing.StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

# linear regression
linear_reg=LinearRegression()
linear_reg.fit(X_train,y_train)
y_test_linear_predict=linear_reg.predict(X_test)
y_train_linear_predict=linear_reg.predict(X_train)

#coefficient and intercept for each column
counter=0
for value in linear_reg.coef_:
    print(df.columns[counter],"coefficient:",value)
    counter=counter+1
print("\n")
print("intercept is",linear_reg.intercept_)

#residual plot of linear regression
plt.scatter(y_train_linear_predict,y_train_linear_predict-y_train,color="red",edgecolor="white",label="Training Data")
plt.scatter(y_test_linear_predict,y_test_linear_predict-y_test,edgecolor="white",label="Testing Data")
plt.legend(loc="upper left")
plt.hlines(y=0,xmin=-50,xmax=100,lw=2)
plt.xlim(-10,50)
plt.xlabel("predicted values")
plt.ylabel("residuals")
plt.show()


#MSE and R^2
print("MSE Train:")
print (mean_squared_error(y_train_linear_predict,y_train))

print("MSE Test:")
print(mean_squared_error(y_test_linear_predict,y_test))
print("\n")
print("R^2 Train:")
print (r2_score(y_train_linear_predict,y_train))

print("R^2 Test:")
print (r2_score(y_test_linear_predict,y_test))
print("\n")



#Ridge Regression
print("Ridge Regression")

# use cross validation to pick the best alpha for Ridge Regression
from sklearn.linear_model import RidgeCV
ridgecv = RidgeCV(alphas=range(1,100),cv=5)
ridgecv.fit(X_train, y_train)
print("\n")
print("under cross validation")
print("the best alpha is",ridgecv.alpha_)
print("\n")


ridge_reg=Ridge(alpha=ridgecv.alpha_)
ridge_reg.fit(X_train,y_train)
y_test_ridge_predict=ridge_reg.predict(X_test)
y_train_ridge_predict=ridge_reg.predict(X_train)

#coefficient and intercept for each column
counter=0
for value in ridge_reg.coef_:
    print(df.columns[counter],"coefficient:",value)
    counter=counter+1
print("\n")
print("intercept is",ridge_reg.intercept_)

#residual plot of ridge regression
plt.scatter(y_train_ridge_predict,y_train_ridge_predict-y_train,color="red",edgecolor="white",label="Training Data")
plt.scatter(y_test_ridge_predict,y_test_ridge_predict-y_test,edgecolor="white",label="Testing Data")
plt.legend(loc="upper left")
plt.hlines(y=0,xmin=-50,xmax=100,lw=2)
plt.xlim(-10,50)
plt.xlabel("predicted values")
plt.ylabel("residuals")
plt.show()


#MSE and R^2
print("MSE Train:")
print (mean_squared_error(y_train_ridge_predict,y_train))

print("MSE Test:")
print(mean_squared_error(y_test_ridge_predict,y_test))
print("\n")
print("R^2 Train:")
print (r2_score(y_train,y_train_ridge_predict))

print("R^2 Test:")
print (r2_score(y_test,y_test_ridge_predict))
print("\n")


#Lasso Regression
print("Lasso Regression")

# use cross validation to pick the best alpha for Lasso Regression
from sklearn.linear_model import LassoCV
lassocv = LassoCV(alphas=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0],cv=5)
lassocv.fit(X_train, y_train)
print("\n")
print("under cross validation")
print("the best alpha is",lassocv.alpha_)
print("\n")



lasso_reg=Lasso(alpha=lassocv.alpha_)

lasso_reg.fit(X_train,y_train)
y_test_lasso_predict=lasso_reg.predict(X_test)
y_train_lasso_predict=lasso_reg.predict(X_train)

#coefficient and intercept for each column
counter=0
for value in lasso_reg.coef_:
    print(df.columns[counter],"coefficient:",value)
    counter=counter+1
print("\n")
print("intercept is",lasso_reg.intercept_)


#residual plot of ridge regression
plt.scatter(y_train_lasso_predict,y_train_lasso_predict-y_train,color="red",edgecolor="white",label="Training Data")
plt.scatter(y_test_lasso_predict,y_test_lasso_predict-y_test,edgecolor="white",label="Testing Data")
plt.legend(loc="upper left")
plt.hlines(y=0,xmin=-50,xmax=100,lw=2)
plt.xlim(-10,50)
plt.xlabel("predicted values")
plt.ylabel("residuals")
plt.show()


#MSE and R^2
print("MSE Train:")
print (mean_squared_error(y_train_lasso_predict,y_train))

print("MSE Test:")
print(mean_squared_error(y_test_lasso_predict,y_test))
print("\n")

print("R^2 Train:")
print (r2_score(y_train_lasso_predict,y_train))


print("R^2 Test:")
print (r2_score(y_test_lasso_predict,y_test))
print("\n")


print("My name is Yi Zhou")
print("My NetID is: yizhou16")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")