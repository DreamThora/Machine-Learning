from email.header import Header
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import glob
from scipy import stats
import datetime as dt

# Load data from csv 3 files
# acceleration.txt, heartrate.txt, labeled_sleep.txt
ACC = pd.read_csv("C:/Vs code file/ML/Ac8/acceleration.txt", sep = ' ',names=['timedelta', 'accX', 'accY', 'accZ'])
HeartR = pd.read_csv("C:/Vs code file/ML/Ac8/heartrate.txt", sep = ',',names=['timedelta', 'heartrate'])
SleepL = pd.read_csv("C:/Vs code file/ML/Ac8/labeled_sleep.txt", sep = ' ',names=['timedelta', 'sleep'])
#C:\Vs code file\ML\Ac8

#check timedelta,min,max of acc,hr,slp
ACC_max_date = ACC["timedelta"].max()
ACC_min_date = ACC["timedelta"].min()

HR_max_date = HeartR["timedelta"].max()
HR_min_date = HeartR["timedelta"].min()

Slp_max_date = SleepL["timedelta"].max()
Slp_min_date = SleepL["timedelta"].min()

#จารให้หา start_timedelta, end_timedelta ส่วนตัวคิดว่าเป็น min กับ max ตามลำดับ

ACC_new = ACC[(ACC["timedelta"]> ACC_min_date) & (ACC["timedelta"] < ACC_max_date) & (HeartR["timedelta"]> HR_min_date) & (HeartR["timedelta"] < HR_max_date) &(SleepL["timedelta"]> Slp_min_date) & (SleepL["timedelta"] < Slp_max_date)]
HeartR_new = HeartR[(ACC["timedelta"]> ACC_min_date) & (ACC["timedelta"] < ACC_max_date) & (HeartR["timedelta"]> HR_min_date) & (HeartR["timedelta"] < HR_max_date) &(SleepL["timedelta"]> Slp_min_date) & (SleepL["timedelta"] < Slp_max_date)]
SleepL_new = SleepL[(ACC["timedelta"]> ACC_min_date) & (ACC["timedelta"] < ACC_max_date) & (HeartR["timedelta"]> HR_min_date) & (HeartR["timedelta"] < HR_max_date) &(SleepL["timedelta"]> Slp_min_date) & (SleepL["timedelta"] < Slp_max_date)]



# print("-----before convert datetime and round and average to 1s-----")
# print(ACC_new.loc[0:10])
# ------------ Rounding ACC (Rounding to 1 sec) -------------------------------
# Convert to datetime and round to second,


ACC_new['timedelta'] = pd.DataFrame(pd.to_timedelta(ACC_new['timedelta'], 'seconds').round('1s'))

df_acc_X = ACC_new.groupby('timedelta')['accX'].mean()
df_acc_Y = ACC_new.groupby('timedelta')['accY'].mean()
df_acc_Z = ACC_new.groupby('timedelta')['accZ'].mean()

ACC_new2 = pd.concat([df_acc_X, df_acc_Y, df_acc_Z], axis=1)
ACC_new2 = ACC_new2.reset_index()
ACC_new2['timedelta'] = ACC_new2['timedelta'] - ACC_new2['timedelta'].min()

# print("-----after convert datetime and round and average to 1s-----")
# print(ACC_new2)



# ------------ Rounding Heart Rate (Rounding to 1 sec) -------------------------------
HeartR_new['timedelta'] = pd.DataFrame(pd.to_timedelta(HeartR_new['timedelta'],'seconds').round('1s'))
# Resampling every 1s with median with ffill
resample_rule = '1s'
HeartR_new2 = HeartR_new.set_index('timedelta').resample(resample_rule,).median().ffill()
HeartR_new2 = HeartR_new2.reset_index()
HeartR_new2['timedelta'] = HeartR_new2['timedelta'] - HeartR_new2['timedelta'].min()


# ------------ Rounding Sleep Label (Rounding to 1 sec) -------------------------------
SleepL_new['timedelta'] = pd.DataFrame(pd.to_timedelta(SleepL_new['timedelta'],'seconds').round('1s'))
# Resampling every 1s with median with ffill
resample_rule = '1s'
SleepL_new2 = SleepL_new.set_index('timedelta').resample(resample_rule,).median().ffill()
SleepL_new2 = SleepL_new2.reset_index()
SleepL_new2['timedelta'] = SleepL_new2['timedelta'] - SleepL_new2['timedelta'].min()



#8.1E merge all data
df = []
df = pd.merge_asof(ACC_new2,HeartR_new2,on='timedelta')
df = pd.merge_asof(df, SleepL_new2, on = 'timedelta')

HeartR_new2['heartrate'].fillna(HeartR_new2.median())
SleepL_new2['sleep'].fillna(0)

df.drop(columns = ['timedelta'],inplace = True)

# Standardized data
feature_columns = ['accX', 'accY', 'accZ', 'heartrate']
label_columns = ['sleep']


# Standardized data
standard_scaler = StandardScaler()
feature_columns = ['accX', 'accY', 'accZ', 'heartrate']
label_columns = ['sleep']
df_feature = df[feature_columns] #standardized data of df_feature
df_feature = pd.DataFrame(standard_scaler.fit_transform(df_feature.values),index = df_feature.index,columns=df_feature.columns)
print(df_feature)
df_label = df[label_columns]
print(df_label)

# Visualize signals
df_feature.plot()
plt.show()
df_label.plot()
plt.show()