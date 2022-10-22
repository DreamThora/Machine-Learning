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

ACC = pd.read_csv("C:/Vs code file/Machine Learning/Machine-Learning/Ac9/acceleration.txt", sep = ' ',names=['timedelta', 'accX', 'accY', 'accZ'])
HeartR = pd.read_csv("C:/Vs code file/Machine Learning/Machine-Learning/Ac9/heartrate.txt", sep = ',',names=['timedelta', 'heartrate'])
SleepL = pd.read_csv("C:/Vs code file/Machine Learning/Machine-Learning/Ac9/labeled_sleep.txt", sep = ' ',names=['timedelta', 'sleep'])
#C:\Vs code file\ML\Ac8

#check timedelta,min,max of acc,hr,slp
ACC_max_date = ACC["timedelta"].max()
ACC_min_date = ACC["timedelta"].min()

HR_max_date = HeartR["timedelta"].max()
HR_min_date = HeartR["timedelta"].min()

Slp_max_date = SleepL["timedelta"].max()
Slp_min_date = SleepL["timedelta"].min()


print("ACC max : {0} ACC min : {1}",ACC_max_date,ACC_min_date)
print("HR max : {0} HR min : {1}",HR_max_date,HR_min_date)
print("Slp max : {0} Slp min : {1}",Slp_max_date,Slp_min_date)



ACC_new = ACC[(ACC["timedelta"]> ACC_min_date) & (ACC["timedelta"] < ACC_max_date) & (ACC["timedelta"]> HR_min_date) & (ACC["timedelta"] < HR_max_date) &(ACC["timedelta"]> Slp_min_date) & (ACC["timedelta"] < Slp_max_date)]
HeartR_new = HeartR[(HeartR["timedelta"]> ACC_min_date) & (HeartR["timedelta"] < ACC_max_date) & (HeartR["timedelta"]> HR_min_date) & (HeartR["timedelta"] < HR_max_date) &(HeartR["timedelta"]> Slp_min_date) & (HeartR["timedelta"] < Slp_max_date)]
SleepL_new = SleepL[(SleepL["timedelta"]> ACC_min_date) & (SleepL["timedelta"] < ACC_max_date) & (SleepL["timedelta"]> HR_min_date) & (SleepL["timedelta"] < HR_max_date) &(SleepL["timedelta"]> Slp_min_date) & (SleepL["timedelta"] < Slp_max_date)]

print("-----before convert datetime and round and average to 1s-----")
print(ACC_new)


# ------------ Rounding ACC (Rounding to 1 sec) -------------------------------
# Convert to datetime and round to second,
ACC_new['timedelta'] = pd.DataFrame(pd.to_timedelta(ACC_new['timedelta'], 'seconds').round('1s'))

df_acc_X = ACC_new.groupby('timedelta')['accX'].mean()
df_acc_Y = ACC_new.groupby('timedelta')['accY'].mean()
df_acc_Z = ACC_new.groupby('timedelta')['accZ'].mean()

ACC_new2 = pd.concat([df_acc_X, df_acc_Y, df_acc_Z], axis=1)
ACC_new2 = ACC_new2.reset_index()
ACC_new2['timedelta'] = ACC_new2['timedelta'] - ACC_new2['timedelta'].min()

print("-----after convert datetime and round and average to 1s-----")
print(ACC_new2)

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
df = pd.merge_asof(ACC_new2,HeartR_new2,on='timedelta',direction='nearest')
df = pd.merge_asof(df, SleepL_new2, on = 'timedelta',direction='nearest')

HeartR_new2['heartrate'].fillna(HeartR_new2.median())
SleepL_new2['sleep'].fillna(0)


df.drop(columns = ['timedelta'],inplace = True)
print(df)
# Standardized data
feature_columns = ['accX', 'accY', 'accZ', 'heartrate']
label_columns = ['sleep']


# Standardized data
standard_scaler = StandardScaler()
feature_columns = ['accX', 'accY', 'accZ', 'heartrate']
label_columns = ['sleep']
df_feature = df[feature_columns] #standardized data of df_feature
df_feature = pd.DataFrame(standard_scaler.fit_transform(df_feature.values),index = df_feature.index,columns=df_feature.columns)
# print(df_feature)
df_label = df[label_columns]
# print(df_label)

# Visualize signals
# df_feature.plot()
# plt.show()
# df_label.plot()
# plt.show()

# ------------ 1D to 3D feature-------------------------------
# set sliding window parameter

df_feature3D=[[]]
df_feature3D = np.array(df_feature3D)
df_label_new=[]
df_label_new = np.array(df_label_new)
slidingW = 100
Stride_step = 5
n_features = 4 #number of colums form df_feature

for t in range( 0 , len(df_feature), Stride_step ):
    F3d = df_feature[t:t+slidingW]
    print("F3d: ",F3d)
    print(df_feature3D)
    df_feature3D=np.concatenate((df_feature3D,F3d),axis=1)
    df_feature3D.reshape(slidingW, n_features , 1)
    Labels = stats.mode(df_label[t : t+slidingW])
    df_label_new.append(Labels)

# ------------ Train-Test-Split 2D features -------------------------------
x_train, x_test, y_train, y_test = train_test_split( df_feature, df_label)

# ------------ Train-Test-Split 3D features -------------------------------
x3D_train, x3D_test, y3D_train, y3D_test = train_test_split( df_feature3D , df_label_new)

