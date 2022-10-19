import csv
from multiprocessing.sharedctypes import Value
from optparse import Values
from os import stat
from pydoc import describe
from re import T
import string
from tkinter import Variable
from tracemalloc import Statistic
from turtle import title
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.feature_selection import chi2
data = pd.read_csv("https://raw.githubusercontent.com/srivatsan88/YouTubeLI/master/dataset/churn_data_st.csv",sep=",")
print(data.shape)
data.drop('customerID', inplace=True, axis=1)
print(data.info())
#found na at TotalCharges
#fillna as mean
data["TotalCharges"] = data["TotalCharges"].fillna(data["TotalCharges"].mode())
print(data.info())
newdata = data.copy()
newdata.drop(['gender','SeniorCitizen','Contract','PaperlessBilling','Churn'], inplace=True, axis=1)
print(newdata.info())
corr = newdata.corr()
print("corr")
print(corr)
#ax = sns.heatmap(corr, vmin=0, vmax=1)
#plt.show()
lower = pd.DataFrame(np.tril(corr, -1),columns = corr.columns)
print("lower")
print(lower)
to_drop = [column for column in lower if any(lower[column] > 0.6)]
newdata.drop(to_drop, inplace=True, axis=1)
print(newdata)
print(newdata.describe())
newnewdata = data.copy()
print(newnewdata)
newnewdata.drop(['gender','tenure','ServiceCount','SeniorCitizen','Contract','PaperlessBilling','MonthlyCharges','TotalCharges'],inplace=True,axis=1)
print(newnewdata)
catagoryToLebel=preprocessing.LabelEncoder()
Output = catagoryToLebel.fit_transform(newnewdata["Churn"])
newnewdata['Churn'] = Output
print("output")
print(Output)
Variables = data.copy()
Variables.drop(['tenure','ServiceCount','SeniorCitizen','MonthlyCharges','TotalCharges','Churn'],inplace=True,axis=1)
Variables["gender"] = catagoryToLebel.fit_transform(Variables["gender"])
Variables["Contract"] = catagoryToLebel.fit_transform(Variables["Contract"])
Variables["PaperlessBilling"] = catagoryToLebel.fit_transform(Variables["PaperlessBilling"])
print(Variables)
Chi_table = chi2(Variables,Output)
print(Chi_table)
p_value = Chi_table[1]
print(p_value)
lower = pd.DataFrame(np.tril(p_value, -1),columns = Variables.columns)
print(lower)
to_drop = [column for column in lower if any(lower[column] > 0.05)] #5% significant
Variables.drop(to_drop, inplace=True, axis=1)
print(Variables)
result = newdata.join(Variables)
result = result.join(newnewdata['Churn'])
print(result)