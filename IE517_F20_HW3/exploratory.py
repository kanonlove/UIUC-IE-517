import pandas as pd
import seaborn as sns
import sys
import numpy as np
from pandas import DataFrame
from datetime import datetime
import pylab
import scipy.stats as stats
import matplotlib.pyplot as plt;

df=pd.read_csv("HY_Universe_corporate bond.csv")

print(df.head())
print(df.describe())
print("\n")

# COUPON
# obviously there are some outliers in coupon
print("coupon EDA")
plt.plot(df.iloc[:,9])
plt.xlabel("coupon")
plt.ylabel("coupon value")
plt.show()
print("there are some outliers in the coupon data")
print("\n")

#I replace the 999 value with the mean value of the coupon

#I can also use replace function in pandas
#df.replace(to_replace=999,value=np.mean(df.iloc[:,9])) 
i=0
while i<=2720:
    if df.iloc[i,9]==999:
        df.iloc[i,9]=np.mean(df.iloc[:,9])
        print(i," is an outlier")
    i=i+1


print("\n") 
print("after replacement")
plt.boxplot(df.iloc[:,9])
plt.xlabel("coupon")
plt.ylabel("coupon value")
plt.show()
print("\n")

plt.hist(df.iloc[:,9],bins=[0,2.5,5,7.5,10,12.5,15,17.5,20],edgecolor="black")
plt.xlabel("coupon value")
plt.ylabel("coupon count")
plt.show()

print("the mean value of the coupon is",np.mean(df.iloc[:,9]))
print("the variance of the coupon is",np.var(df.iloc[:,9]))
print("the std of the coupon is",np.std(df.iloc[:,9]))
print("\n")


#ISSUED AMOUNT
print("issued amount EDA")

plt.boxplot(df.iloc[:,10])
plt.xlabel("issued amount")
plt.ylabel("issued amount")
plt.show()
print("the mean value of the coupon is",np.mean(df.iloc[:,10])/100000000,"x10^9")
print("the std of the coupon is",np.std(df.iloc[:,10])/100000000,"x10^9")
print("\n")


"""
#TICKER
print("ticker EDA")
print(df.iloc[:,1])
print("\n")
ticker_type=list()

for value in df.iloc[:,1]:
    ticker_type.append(value)
ticker_type=list(set(ticker_type))
ticker_type.sort()

print(ticker_type)
print("\n")


# Maturity Type
print("maturity type EDA")
print(df.iloc[:,11])

maturity_type=list()
for value in df.iloc[:,11]:
    maturity_type.append(value)

maturity_type=list(set(maturity_type))
maturity_type.sort()
print(maturity_type)
print("\n")

# Industry

print("industry EDA")
print(df.iloc[:,14])
print("\n")
industry_type=list()

for value in df.iloc[:,14]:
    industry_type.append(value)
industry_type=list(set(industry_type))
industry_type.sort()

print(industry_type)
print("\n")

"""


#BOND TERM TO MATURITY   
print("BOND TERM TO MATURITY ")
def timeconvert(s):
    k=s.split("/")
    return datetime(int(k[2]), int(k[0]), int(k[1])) 

issue_date=df.iloc[:,2]
maturity_date=df.iloc[:,3]

issue_date_time=list()
maturity_date_time=list()

for value in issue_date:
    
    value=timeconvert(value)
    issue_date_time.append(value)
for value in maturity_date:
    if value=="Nan Field Not Applicable":
        value="N/A"
    else:
        value=timeconvert(value)
    maturity_date_time.append(value)

i=0
maturity_term=list()
for i in range(0,2721):
    if maturity_date_time[i]!="N/A":
        maturity_term.append(round((maturity_date_time[i]-issue_date_time[i]).days/365,3))


plt.hist(maturity_term,edgecolor="black",bins=[0,5,10,15,20,25,30,35,40])
plt.xlabel("years")
plt.ylabel("bond numbers")
plt.show()

#CALLABLE
print("callable or not")
i=0
false_number=0
true_number=0
for value in  df.iloc[:,4]:
    if value=="Nan":
        df.iloc[i,4]=False
        false_number=false_number+1
    else:
        df.iloc[i,4]=True
        true_number=true_number+1
    i=i+1


plt.bar(["Callable","Not Callable"],height=[true_number,false_number],width=0.4,edgecolor="black")
plt.xlabel("Whether it is callable")
plt.ylabel("Number")
plt.show()
print("\n")


##qq plot between coupon and normal distribution
print("accumulative distribution function of coupon")
coupon=df.iloc[:,9]


sorted_coupon = np.sort(coupon)
coupon_y = np.arange(len(sorted_coupon))/float(len(sorted_coupon))
plt.plot(sorted_coupon, coupon_y)
plt.xlabel("accumulative distribution function of Coupon")
plt.show()


stats.probplot(coupon, dist="norm", plot=plt)
plt.xlabel("QQ plot of Coupon to normal distribution")
plt.show()
print("\n")

#relationship between coupon and maturity at issued months
print("coupon and maturity at issued months")
coupon = df.iloc[:,9]
maturity_at_issued_months = df.iloc[:,13]
plt.scatter(coupon, maturity_at_issued_months)

plt.xlabel("coupon")
plt.ylabel(("maturity_at_issued_months"))
plt.show()
print("\n")


#heat map
print("heat map of all the data")
cormat = DataFrame(df.corr())
#visualize correlations using heatmap
plt.pcolor(cormat)
plt.show()
print("\n")

print("My name is Yi Zhou")
print("My NetID is: yizhou16")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")
    








