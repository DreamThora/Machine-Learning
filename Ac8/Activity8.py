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
from sklearn.model_selection import train_test_split
from sklearn import model_selection

# Load data from csv 3 files
# acceleration.txt, heartrate.txt, labeled_sleep.txt
ACC = pd.read_csv("C:/Vs code file/Machine Learning/Machine-Learning/Ac8/acceleration.txt", sep = ' ',names=['timedelta', 'accX', 'accY', 'accZ'])
HeartR = pd.read_csv("C:/Vs code file/Machine Learning/Machine-Learning/Ac8/heartrate.txt", sep = ',',names=['timedelta', 'heartrate'])
SleepL = pd.read_csv("C:/Vs code file/Machine Learning/Machine-Learning/Ac8/labeled_sleep.txt", sep = ' ',names=['timedelta', 'sleep'])
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



#จารให้หา start_timedelta, end_timedelta ส่วนตัวคิดว่าเป็น min กับ max ตามลำดับ

# ACC_new = ACC[(ACC["timedelta"]> ACC_min_date) & (ACC["timedelta"] < ACC_max_date) & (HeartR["timedelta"]> HR_min_date) & (HeartR["timedelta"] < HR_max_date) &(SleepL["timedelta"]> Slp_min_date) & (SleepL["timedelta"] < Slp_max_date)]
# HeartR_new = HeartR[(ACC["timedelta"]> ACC_min_date) & (ACC["timedelta"] < ACC_max_date) & (HeartR["timedelta"]> HR_min_date) & (HeartR["timedelta"] < HR_max_date) &(SleepL["timedelta"]> Slp_min_date) & (SleepL["timedelta"] < Slp_max_date)]
# SleepL_new = SleepL[(ACC["timedelta"]> ACC_min_date) & (ACC["timedelta"] < ACC_max_date) & (HeartR["timedelta"]> HR_min_date) & (HeartR["timedelta"] < HR_max_date) &(SleepL["timedelta"]> Slp_min_date) & (SleepL["timedelta"] < Slp_max_date)]
ACC_new = ACC[(ACC["timedelta"]> ACC_min_date) & (ACC["timedelta"] < ACC_max_date) & (ACC["timedelta"]> HR_min_date) & (ACC["timedelta"] < HR_max_date) &(ACC["timedelta"]> Slp_min_date) & (ACC["timedelta"] < Slp_max_date)]
HeartR_new = HeartR[(HeartR["timedelta"]> ACC_min_date) & (HeartR["timedelta"] < ACC_max_date) & (HeartR["timedelta"]> HR_min_date) & (HeartR["timedelta"] < HR_max_date) &(HeartR["timedelta"]> Slp_min_date) & (HeartR["timedelta"] < Slp_max_date)]
SleepL_new = SleepL[(SleepL["timedelta"]> ACC_min_date) & (SleepL["timedelta"] < ACC_max_date) & (SleepL["timedelta"]> HR_min_date) & (SleepL["timedelta"] < HR_max_date) &(SleepL["timedelta"]> Slp_min_date) & (SleepL["timedelta"] < Slp_max_date)]


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
df_feature.plot()
# plt.show()
df_label.plot()
# plt.show()


#train test split
rseed=42
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(df_feature, df_label, test_size=0.3, random_state=rseed)

#Model Traing Parameter
# Create SVC model
c_val = 100 
gmm =0.1
d = 2
# Model initialize
svc_lin = SVC(kernel='linear', C=c_val)
svc_rbf = SVC(kernel='rbf', C=c_val, gamma=gmm)
svc_poly = SVC(kernel='poly', C=c_val, degree = d)
# Model Training
svc_rbf_pred = svc_rbf.fit(X_train, Y_train)
svc_poly = svc_poly.fit(X_train, Y_train)

# Model Testing (Predict)
svc_rbf_pred = svc_rbf.predict(X_test)
svc_poly_pred = svc_poly.predict(X_test)

# Model Confusion Matrix of SVC_rbf, SVC_poly
confusion_matrix(Y_test,svc_rbf_pred)

# Model Classification Report of SVC_rbf, SVC_poly
classification_report(Y_test,svc_rbf_pred)


print('Confusion Matrix of SVC RBF: ')
print(confusion_matrix(Y_test, svc_rbf_pred))
print('Classification Report of SVC RBF: ')
print(classification_report(Y_test, svc_rbf_pred))

print('Confusion Matrix of SVC Poly: ')
print(confusion_matrix(Y_test, svc_poly_pred))
print('Classification Report of SVC Poly: ')
print(classification_report(Y_test, svc_poly_pred))


#Create Model Parameter Dictionary for SVC
C_list = [0.1, 1.0, 10.0, 100.0, 200.0, 500.0]
Gamma_list = [0.01, 0.1, 1.0, 10]
d_list = [2, 3]


#Create Model Parameter Dictionary for SVC



#swap between poly and rbf cause poly take around 8-10hr
kernel = ['poly'] #['rbf','poly']
C_list = [0.1, 1.0, 10.0, 100.0, 200.0, 500.0]
Gamma_list = [0.01, 0.1, 1.0, 10]
d_list = [2, 3]
params = dict(kernel = kernel,C = C_list,gamma = Gamma_list,degree = d_list)
# Perform GridsearchCV() for each classification model
grid = GridSearchCV( estimator=  SVC(), n_jobs = 8, verbose = 10, scoring = 'accuracy', cv = 2, param_grid = params)
grid_result = grid.fit(X_train, Y_train)
print('Best params: ',grid_result.best_params_)
print('Best score: ', grid_result.best_score_)
mean = grid_result.cv_results_['mean_test_score']
std = grid_result.cv_results_['std_test_score']
param = grid_result.cv_results_['params']

bar_mean = []
bar_std = []
bar_param = []
for mean, stdev, param in zip(mean, std, param):
    print("%f (%f) with: %r" % (mean, stdev, param))
    bar_mean.append(mean)
    bar_std.append(stdev)
    bar_param.append("C_list : "+ str(param['C']) + ", G_list : " + str(param['gamma']) + ", D_list : " + str(param['degree']))

x = np.arange(len(bar_mean))
w = 0.5
fig, ax = plt.subplots()
fig = plt.title('Kernel : poly')
rect1 = plt.bar(x-w/2,bar_mean,w,color = 'b')
rect2 = plt.bar(x+w/2,bar_std,w,color = 'r')
ax.set_xticks(x, labels= bar_param,fontsize=6,rotation = 90)
plt.subplots_adjust(bottom=0.20)
plt.show()