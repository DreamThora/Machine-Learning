import csv
from multiprocessing.sharedctypes import Value
from optparse import Values
from os import stat
from pydoc import describe
import string
from tkinter import Variable
from tracemalloc import Statistic
from turtle import title
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
# explore
data = pd.read_csv("Data_example.csv",encoding="ISO-8859-1")
# explore
print("########explore########")
print(data)
print(data.describe())
print("########explore########")
# clean
print("########clean########")
data = data.dropna(thresh=2)
data = data.drop_duplicates()
data.replace({r'[^\x00-\x7E]+':''},regex=True,inplace=True)
data.replace({r'[A]+': np.nan}, regex=True, inplace=True)
data["X"] = pd.to_numeric(data["X"], errors='coerce')
Xmedian = int(np.floor(np.nanmedian(data["X"])))
data["Z"].fillna(method='bfill', inplace=True)
data["X"] = data["X"].fillna(Xmedian)
data["X"] = data["X"].astype("Int64")
data["Y"] = pd.to_numeric(data["Y"], errors='coerce')
data["Y"] = data["Y"].fillna(np.mean(data["Y"]))
data["Y"] = data["Y"].astype("Float64")
data["Z"] = data["Z"].astype("string")

print(data)
print(data.describe())
print("########clean########")
# clean
# transform

scaled_features = data.copy()
featuresX = scaled_features[["X"]]
featuresY = scaled_features[["Y"]]
standardScaler = preprocessing.StandardScaler()
minMaxScaler = preprocessing.MinMaxScaler()
scaled_features[["X"]]= minMaxScaler.fit_transform(featuresX,Values)
scaled_features[["Y"]]= minMaxScaler.fit_transform(featuresY,Values)

print(scaled_features)
#sns.boxplot(y= scaled_features["X"])
#plt.show()
#Xoutlier
Q1X=scaled_features["X"].quantile(0.25)
Q3X=scaled_features["X"].quantile(0.75)
IQRX = Q3X-Q1X
upperLimitX=Q3X+1.5*IQRX
lowerLimitX=Q1X-1.5*IQRX
new_scaled_featuresXupper = scaled_features[scaled_features['X'] < upperLimitX]
new_scaled_featuresXlower = new_scaled_featuresXupper[new_scaled_featuresXupper['X'] > lowerLimitX]
new_scaled_featuresXlower.shape
#Xoutlier
#Yourlier
Q1Y=scaled_features["Y"].quantile(0.25)
Q3Y=scaled_features["Y"].quantile(0.75)
IQRY = Q3Y-Q1Y
upperLimitY=Q3Y+1.5*IQRY
lowerLimitY=Q1Y-1.5*IQRY
new_scaled_featuresXYupper = new_scaled_featuresXlower[new_scaled_featuresXlower['Y'] < upperLimitY]
new_scaled_featuresXY = new_scaled_featuresXupper[new_scaled_featuresXupper['Y'] > lowerLimitY]
new_scaled_featuresXY.shape
print(new_scaled_featuresXY)
#Youtlier
removeOutLier = new_scaled_featuresXY.copy()
newfeaturesX = removeOutLier[["X"]]
newfeaturesY = removeOutLier[["Y"]]
standardScaler = preprocessing.StandardScaler()
minMaxScaler = preprocessing.MinMaxScaler()
removeOutLier[["X"]]= minMaxScaler.fit_transform(newfeaturesX,Value)
removeOutLier[["Y"]]= minMaxScaler.fit_transform(newfeaturesY,Value)
reset=removeOutLier.reset_index()
reset.drop('index', inplace=True, axis=1)
newfeaturesZ = reset[["Z"]]
catagoryToLebel=preprocessing.LabelEncoder()
catagoryToOneHot = preprocessing.OneHotEncoder()
print(newfeaturesZ)
labelencoder =catagoryToLebel.fit_transform(newfeaturesZ)
reset["Label"] = labelencoder
onehotencoder = pd.DataFrame(catagoryToOneHot.fit_transform(newfeaturesZ).toarray())
onehotencoder.columns=["bird","cat","dog"]
print(onehotencoder)
complete = reset.join(onehotencoder)




print(complete)


sns.boxplot(y= complete["Y"])
plt.show()