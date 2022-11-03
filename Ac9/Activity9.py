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

# ------------ 1D to 3D feature-------------------------------
# set sliding window parameter
slidingW = 100
Stride_step = 5
n_features = 4 #number of colums form df_feature
df_feature3D = np.array([],ndmin=2)
df_label_new = np.array([])

for t in range(0 , len(df_feature), Stride_step ):
    F3d = np.array(df_feature[t:t+slidingW],ndmin=2)
    # print(F3d[0])
    if len(F3d) <slidingW:
        break
    # print(df_feature3D.shape)
    # print(F3d.shape)
    if df_feature3D.size == 0 :
        df_feature3D = F3d
    else:
        df_feature3D = np.dstack((df_feature3D,F3d))
    Labels = stats.mode(df_label[t : t+slidingW])
    # print(Labels)
    df_label_new = np.append(df_label_new,Labels[0])
    # print(df_feature3D.shape)
    # print(df_label_new.shape)
    # print(df_feature3D)

# df_feature3D = pd.DataFrame(df_feature3D) 
df_feature3D = np.swapaxes(df_feature3D,0,2)
df_feature3D = np.swapaxes(df_feature3D,1,2)
df_feature3D = df_feature3D[..., np.newaxis]
# print(df_feature3D.shape)
# print(df_label_new.shape)

# ------------ Train-Test-Split 2D features -------------------------------
rseed=42
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(df_feature, df_label, test_size=0.3, random_state=rseed)

# ------------ Train-Test-Split 3D features -------------------------------
x3D_train, x3D_test, y3D_train, y3D_test = model_selection.train_test_split(df_feature3D, df_label_new, test_size=0.3, random_state=rseed)

# ------------ NN Architecture parameter -------------------------------
Hidden_Layer_param = (30, 30, 30)
mlp = MLPClassifier(hidden_layer_sizes = Hidden_Layer_param)
# View NN model parameters

# ------------ Training NN using 1D features -------------------------------
mlp.fit(X_train,Y_train)
mlp_pred = mlp.predict(X_test)
print('Confusion Matrix of mlp_pred 1: ')
print(confusion_matrix(Y_test,mlp_pred))
print('Classification Report of mlp_pred 1: ')
print(classification_report(Y_test,mlp_pred))

# ------------ CNN Architecture parameter -------------------------------
# Nlayer (CNN, dense), Nnode, Activation
CNN_L1 = 16
CNN_L2 = 64
CNN_L3 = 128
D_L1 = 512
D_out = 6
Activation = "relu"
Ker_size = (3,3)
Pooling_size = (2,1)
Input_shape = (slidingW, n_features, 1)

# ------------ Create CNN Model -------------------------------

model = models.Sequential()
model.add(Conv2D(CNN_L1, kernel_size=Ker_size, activation=Activation,input_shape=Input_shape,padding='same'))
model.add(MaxPooling2D(pool_size=Pooling_size))
model.add(Dropout(0.4))
model.add(Conv2D(CNN_L2, kernel_size=Ker_size, activation= Activation, padding='same'))
model.add(MaxPooling2D(pool_size= Pooling_size))
model.add(Dropout(0.4))
model.add(Conv2D(CNN_L3, kernel_size=Ker_size, activation= Activation,padding='same'))
model.add(MaxPooling2D(pool_size= Pooling_size))
model.add(Dropout(0.4))
model.add(Flatten())
model.add(Dense(D_L1 , activation= Activation ))
model.add(Dense(D_out, activation='sigmoid'))
model.compile(optimizer='adam', metrics=['accuracy'])
model.summary()


# ------------ Create Optimizer -------------------------------
model.compile(optimizer='adam',loss= tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),metrics=["acc"])

# ------ Train CNN using 3D feature--------------------------------------------
history = model.fit(x3D_train, y3D_train, epochs=50, batch_size=64,validation_data=(x3D_test, y3D_test))

# ------- Test CNN -------------------------------

CNN_pred = np.argmax(model.predict(x3D_test),axis=1)
print(CNN_pred)

# ------------ View Confusion Matrix, Classification Report -------------------------------
print('Confusion Matrix of mlp_pred 2: ')
print(confusion_matrix(y3D_test,CNN_pred))
print('Classification Report of mlp_pred 2: ')
print(classification_report(y3D_test,CNN_pred))
# ------ View History Graph -------------------------------------------
# View Accuracy Graph
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.show()
# View Loss Graph
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.show()