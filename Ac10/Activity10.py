import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.svm import SVC
from sklearn import metrics,model_selection
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import glob
from scipy import stats
import datetime as dt
from sklearn.neural_network import MLPClassifier
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.layers import Conv2D, MaxPooling2D,Dropout,Flatten,Dense

ACC = pd.read_csv("acceleration.txt", sep = ' ',names=['timedelta', 'accX', 'accY', 'accZ'])
HeartR = pd.read_csv("heartrate.txt", sep = ',',names=['timedelta', 'heartrate'])
SleepL = pd.read_csv("labeled_sleep.txt", sep = ' ',names=['timedelta', 'sleep'])
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

# print("-----before convert datetime and round and average to 1s-----")
# print(ACC_new)


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

SleepL_new2.replace({-1:0},inplace=True)
SleepL_new2['sleep'].fillna(0)

#8.1E merge all data
df = []
df = pd.merge_asof(ACC_new2,HeartR_new2,on='timedelta',direction='nearest')
df = pd.merge_asof(df, SleepL_new2, on = 'timedelta',direction='nearest')

HeartR_new2['heartrate'].fillna(HeartR_new2.median())


df.drop(columns = ['timedelta'],inplace = True)
# print(df)
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
# print("df_label")
# print(df_label)
# Visualize signals
# df_feature.plot()
# plt.show()
# df_label.plot()
# plt.show()


# acceleration.txt, heartrate.txt, labeled_sleep.txt
# Rounding ACC (Rounding to 1 sec)
ACC_new['timedelta'] = pd.DataFrame(pd.to_timedelta(ACC_new['timedelta'], 'seconds').round('1s'))

df_acc_X = ACC_new.groupby('timedelta')['accX'].mean()
df_acc_Y = ACC_new.groupby('timedelta')['accY'].mean()
df_acc_Z = ACC_new.groupby('timedelta')['accZ'].mean()

# ACC Average rounding duplicated time
ACC_new2 = pd.concat([df_acc_X, df_acc_Y, df_acc_Z], axis=1)
ACC_new2 = ACC_new2.reset_index()
ACC_new2['timedelta'] = ACC_new2['timedelta'] - ACC_new2['timedelta'].min()

# Rounding Heart Rate (Rounding to 1 sec)
HeartR_new['timedelta'] = pd.DataFrame(pd.to_timedelta(HeartR_new['timedelta'],'seconds').round('1s'))

# Resampling every 1s with median with ffill
resample_rule = '1s'
HeartR_new2 = HeartR_new.set_index('timedelta').resample(resample_rule,).median().ffill()
HeartR_new2 = HeartR_new2.reset_index()
HeartR_new2['timedelta'] = HeartR_new2['timedelta'] - HeartR_new2['timedelta'].min()

# Rounding Sleep Label (Rounding to 1 sec) 
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


df.drop(columns = ['timedelta'],inplace = True)
# print(df)
# Standardized data
feature_columns = ['accX', 'accY', 'accZ', 'heartrate']
label_columns = ['sleep']

# print(df_feature)
# print(df_label.describe())

# ------------Simple Moving Average (SMA) ------------------------------
df_feature_SMA = pd.DataFrame(columns=['accX','accY','accZ','heartrate'])
df_feature_SMA['accX'] = df_feature['accX'].rolling(5, min_periods=1).mean()
df_feature_SMA['accY'] = df_feature['accY'].rolling(5, min_periods=1).mean()
df_feature_SMA['accZ'] = df_feature['accZ'].rolling(5, min_periods=1).mean()
df_feature_SMA['heartrate'] = df_feature['heartrate'].rolling(5, min_periods=1).mean()

# ------------ Train-Test-Split 2D features -------------------------------
# set sliding window parameter
slidingW = 100 #จ ํานวน row
Stride_step = 5
df_feature2D = np.array([],ndmin=2)
df_label_new = np.array([])
df_feature2D_T = np.array([],ndmin=2)
for t in range( 0 , len(df_feature), Stride_step ):
    F2d= np.array(df_feature[t:t+slidingW],ndmin=2)
    if F2d.size < slidingW*4:
        break
    F2d.reshape(slidingW,4)
    F2d_T = np.transpose(F2d)
    if df_feature2D_T.size == 0 :
        df_feature2D_T = F2d_T
    else:
        df_feature2D_T = np.dstack((df_feature2D_T,F2d_T))
    Labels = stats.mode(df_label[t : t+slidingW])
    df_label_new = np.append(df_label_new,Labels[0])
df_feature2D_T = np.swapaxes(df_feature2D_T,0,2)
# df_feature2D_T = np.swapaxes(df_feature2D_T,1,2)

#------------ Train-Test-Split 2D features no transpose-------------------------------
rseed=42
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(df_feature, df_label, test_size=0.3, random_state=rseed)

print(df_feature2D_T.shape)
print(df_label_new.shape)
# ------------ Train-Test-Split 3D features with transpose -------------------------------
x2_train, x2_test, y2_train, y2_test = model_selection.train_test_split(df_feature2D_T, df_label_new, test_size=0.3, random_state=rseed)