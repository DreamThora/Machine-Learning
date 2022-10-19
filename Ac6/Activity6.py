from math import floor
from operator import truediv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn import model_selection
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn import metrics
import pandas_datareader.data as web
from time import time
from scipy.stats import zscore

from sklearn.cluster import KMeans

#Read Stock data form web
stk_tickers = ['MSFT', 'IBM', 'GOOGL']
ccy_tickers = ['DEXJPUS', 'DEXUSUK']
idx_tickers = ['SP500', 'DJIA', 'VIXCLS']
stk_data = web.DataReader(stk_tickers, 'yahoo')
ccy_data = web.DataReader(ccy_tickers, 'fred')
idx_data = web.DataReader(idx_tickers, 'fred')

# ccy_data = ccy_data.dropna(thresh=2)
# idx_data = idx_data.dropna(thresh=2)

ccy_data=ccy_data.fillna(ccy_data.mode())
idx_data["DJIA"] = idx_data["DJIA"].fillna(idx_data["DJIA"].mode())
idx_data.replace({object:np.nan},regex=True,inplace=True)
idx_data = idx_data.fillna(idx_data.mode())


#Select column
base = stk_data.loc[:, ('Adj Close', 'MSFT')]
X1 = stk_data.loc[:, ('Adj Close', ('GOOGL', 'IBM'))]
X2 = ccy_data
X3 = idx_data
X3.fillna(X3.median(),inplace=True)
#z score
scaler = preprocessing.StandardScaler()
X1 = pd.DataFrame(scaler.fit_transform(X1.values),index = X1.index,columns=X1.columns)
X2 = pd.DataFrame(scaler.fit_transform(X2.values),index = X2.index,columns=X2.columns)
X3 = pd.DataFrame(scaler.fit_transform(X3.values),index = X3.index,columns=X3.columns)
print(X1)
print(X2)
print(X3)
#Calculate ความแตกต่างของค่า ราคา 'Adj Close', 'MSFT’)ย้อนหลัง backHisotry วัน
backHistory = [30, 45, 60, 90, 180, 240] #-> ทดลองหยิบ 3 ค่า 3 รูปแบบ เพื่อดูระยะเวลาการดูค่าข้อมูลย้อนหลงัหลายๆแบบและเปรียบเทียบ MSE
BH1, BH2, BH3 = backHistory[1], backHistory[3], backHistory[4]
Y = base. shift(-BH1)
X4_BH1 = base.diff( BH1).shift( - BH1)
X4_BH2 = base.diff( BH2).shift( - BH2)
X4_BH3 = base.diff( BH3).shift( - BH3)
X4 = pd.concat([X4_BH1, X4_BH2, X4_BH3], axis=1)
X4.columns = ['X4_BH1', 'X4_BH2', 'X4_BH3']
#ถ้าไม่ dropna ทำ zscore ไม่ได้
X4.dropna(inplace=True)
X4 = X4.apply(zscore)
#forming data
X = pd.concat([X1, X2, X3, X4], axis=1)
dataset = pd.concat([Y, X], axis=1)

#dropna describe
dataset.dropna(inplace=True)
# print(dataset.describe())

# Assign X, Y (drop datetime index)
Y = dataset.filter([dataset.columns[0]])
X = dataset.drop(columns = [dataset.columns[0]])
Y=Y.reset_index(drop = True)
X=X.reset_index(drop = True)

#corr 
corr = X.corr()
lower = pd.DataFrame(np.tril(corr, -1),columns = corr.columns)
# drop columns if corr value > 0.9
to_drop = [column for column in lower if any(lower[column] > 0.9)]
X.drop(to_drop, inplace=True, axis=1)
# print(X)

#Train / Test Preparation (try 2 Option)
# Option#1
test_size1 = int(np.floor(0.3 * len( X )))
train_size1 = int(np.floor(0.7 * len( X )))
X_train1, X_test1 = X[0:train_size1], X[train_size1:len(X)]
Y_train1, Y_test1 = Y[0:train_size1], Y[train_size1:len(X)]

# Option #2
rseed=42
X_train2, X_test2, Y_train2, Y_test2 = model_selection.train_test_split(X, Y, test_size=0.3, random_state=rseed)

#create model list
regression = { 'LR': LinearRegression(), 'SVR': SVR(), }

#create parameter dictionary for linear regression
fit_intercept = [True, False]
normalize = [True, False]
params_LR = dict( fit_intercept = fit_intercept, normalize = normalize)

# Create Parameter Dictionary for SVR
kernel = ['linear', 'rbf', 'poly']
C_list = [10, 100]
ep_list = [0.1, 1, 5]
gamma = [0.01, 0.1]
degree = [2, 3]
params_SVR = dict( kernel = kernel, C = C_list, epsilon = ep_list, gamma = gamma, degree = degree )
print(X)
#GridSearch for option 1
for EST in regression:
    model = regression[EST]
    if (EST == 'LR'):
        params = params_LR
    else:
        params = params_SVR

    grid = GridSearchCV(estimator=model, n_jobs = 1, verbose = 10, cv = 2, scoring = 'neg_mean_squared_error', param_grid = params )

    grid_result1 = grid.fit(X_train1, Y_train1)

# Show Best Parameters for both models
print('Best params: ',grid_result1.best_params_)
print('Best score: ', grid_result1.best_score_)
best_search_c_1_btw = grid_result1.best_params_['C']
best_search_degree_1_btw = grid_result1.best_params_['degree']
best_search_gamma_1_btw = grid_result1.best_params_['gamma']
best_search_epsilon_1_btw = grid_result1.best_params_['epsilon']
# Show Score for each parameter combination for option1
means1 = grid_result1.cv_results_['mean_test_score']
stds1 = grid_result1.cv_results_['std_test_score']
params1 = grid_result1.cv_results_['params']
bar_mean1_linear = []
bar_stds1_linear = []
bar_mean1_poly = []
bar_stds1_poly = []
bar_mean1_rbf = []
bar_stds1_rbf = []
for mean1, stdev1, param1 in zip(means1, stds1, params1):
    print("%f (%f) with: %r" % (mean1, stdev1, param1))
    if param1['kernel'] == 'linear':
        bar_mean1_linear.append(mean1)
        bar_stds1_linear.append(stdev1)
    elif param1['kernel'] == 'poly': 
        bar_mean1_poly.append(mean1)
        bar_stds1_poly.append(stdev1)
    else:
        bar_mean1_rbf.append(mean1)
        bar_stds1_rbf.append(stdev1)
#ทำ bar ไม่เป็น
#linear
x = np.arange(len(bar_mean1_linear))
print(x)
w = 0.5
fig, ax = plt.subplots()
fig = plt.title('Option #1 Search linear')
rect1 = plt.bar(x-w/2,bar_mean1_linear,w,color = 'b')
rect2 = plt.bar(x+w/2,bar_stds1_linear,w,color = 'r')
plt.show()
#poly
x = np.arange(len(bar_mean1_poly))
print(x)
w = 0.5
fig, ax = plt.subplots()
fig = plt.title('Option #1 Search poly')
rect1 = plt.bar(x-w/2,bar_mean1_poly,w,color = 'b')
rect2 = plt.bar(x+w/2,bar_stds1_poly,w,color = 'r')
plt.show()
#rbf
x = np.arange(len(bar_mean1_rbf))
print(x)
w = 0.5
fig, ax = plt.subplots()
fig = plt.title('Option #1 Search rbf')
rect1 = plt.bar(x-w/2,bar_mean1_rbf,w,color = 'b')
rect2 = plt.bar(x+w/2,bar_stds1_rbf,w,color = 'r')
plt.show()
#GridSearch for option 2
for EST in regression:
    model = regression[EST]
    if (EST == 'LR'):
        params = params_LR
    else:
        params = params_SVR

    grid = GridSearchCV(estimator=model, n_jobs = 1, verbose = 10, cv = 2, scoring = 'neg_mean_squared_error', param_grid = params )

grid_result2 = grid.fit(X_train2, Y_train2)

# Show Best Parameters for both models
print('Best params: ',grid_result2.best_params_)
print('Best score: ', grid_result2.best_score_)
best_search_c_2_btw = grid_result2.best_params_['C']
best_search_degree_2_btw = grid_result2.best_params_['degree']
best_search_gamma_2_btw = grid_result2.best_params_['gamma']
best_search_epsilon_2_btw = grid_result2.best_params_['epsilon']
# Show Score for each parameter combination for option 2
means2 = grid_result2.cv_results_['mean_test_score']
stds2 = grid_result2.cv_results_['std_test_score']
params2 = grid_result2.cv_results_['params']
bar_mean2_linear = []
bar_stds2_linear = []
bar_mean2_poly = []
bar_stds2_poly = []
bar_mean2_rbf = []
bar_stds2_rbf = []
for mean2, stdev2, param2 in zip(means2, stds2, params2):
    print("%f (%f) with: %r" % (mean2, stdev2, param2))
    if param2['kernel'] == 'linear':
        bar_mean2_linear.append(mean2)
        bar_stds2_linear.append(stdev2)
    elif param2['kernel'] == 'poly': 
        bar_mean2_poly.append(mean2)
        bar_stds2_poly.append(stdev2)
    else:
        bar_mean2_rbf.append(mean2)
        bar_stds2_rbf.append(stdev2)
#ทำ bar ไม่เป็น
#linear
x = np.arange(len(bar_mean2_linear))
print(x)
w = 0.5
fig, ax = plt.subplots()
fig = plt.title('Option #2 Search linear')
rect1 = plt.bar(x-w/2,bar_mean2_linear,w,color = 'b')
rect2 = plt.bar(x+w/2,bar_stds2_linear,w,color = 'r')
plt.show()
#poly
x = np.arange(len(bar_mean2_poly))
print(x)
w = 0.5
fig, ax = plt.subplots()
fig = plt.title('Option #2 Search poly')
rect1 = plt.bar(x-w/2,bar_mean2_poly,w,color = 'b')
rect2 = plt.bar(x+w/2,bar_stds2_poly,w,color = 'r')
plt.show()
#rbf
x = np.arange(len(bar_mean2_rbf))
print(x)
w = 0.5
fig, ax = plt.subplots()
fig = plt.title('Option #2 Search rbf')
rect1 = plt.bar(x-w/2,bar_mean2_rbf,w,color = 'b')
rect2 = plt.bar(x+w/2,bar_stds2_rbf,w,color = 'r')
plt.show()

#6.3

# Create Model List
regression = { 'LR': LinearRegression(), 'SVR': SVR(), }

# Create Parameter Dictionary for Linear Regression
fit_intercept = [True, False]
normalize = [True, False]
params_LR = dict( fit_intercept = fit_intercept, normalize = normalize)

# Create Parameter Dictionary for SVR
kernel = ['linear', 'rbf', 'poly']
C_list = list(np.linspace(0.1, 150, 5, dtype = float))
ep_list = list(np.linspace(0.1, 1, 5, dtype = float))
gamma = list(np.linspace(0.01, 0.1, 5, dtype = float))
degree = [2, 3]
params_SVR = dict( kernel = kernel, C = C_list, epsilon = ep_list, gamma = gamma, degree = degree )
# option #1
### LM_pred
Model_LM = LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=False)
LM_pred = Model_LM.fit(X_train1, Y_train1).predict(X_test1)
n_clusters_LM = np.unique(LM_pred)
kmeans_LM = KMeans(n_clusters=n_clusters_LM.size,random_state=0)
clusters_LM = kmeans_LM.fit_predict(X_test1)
#svr

c_val = best_search_c_1_btw
svr_lin = SVR(kernel='linear', C=c_val)
svr_rbf = SVR(kernel='rbf', C=c_val, gamma=best_search_gamma_1_btw)
svr_poly = SVR(kernel='poly', C=c_val, degree=best_search_degree_1_btw)
SVR_Linear = svr_lin.fit(X_train1,Y_train1).predict(X_test1)
SVR_Rbf = svr_rbf.fit(X_train1,Y_train1).predict(X_test1)
SVR_Poly = svr_poly.fit(X_train1,Y_train1).predict(X_test1)

plt.scatter(np.arange(len(Y_train1)),Y_train1, edgecolors='b',alpha=0.75,s=5)
plt.scatter(np.arange(len(Y_test1)),Y_test1, edgecolors='r',alpha=0.75,s=1)
plt.scatter(np.arange(len(LM_pred)),LM_pred, edgecolors='pink',alpha=0.75,s=1)
plt.scatter(np.arange(len(SVR_Linear)),SVR_Linear,edgecolors='green',alpha=0.75,s=1)
plt.scatter(np.arange(len(SVR_Poly)),SVR_Poly, edgecolors='m',alpha=0.75,s=1)
plt.scatter(np.arange(len(SVR_Rbf)),SVR_Rbf, edgecolors='orange',alpha=0.75,s=1)
plt.show()

# option #2
### LM_pred
Model_LM = LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=False)
LM_pred = Model_LM.fit(X_train2, Y_train2).predict(X_test2)
n_clusters_LM = np.unique(LM_pred)
kmeans_LM = KMeans(n_clusters=n_clusters_LM.size,random_state=0)
clusters_LM = kmeans_LM.fit_predict(X_test2)
#svr
c_val = best_search_c_2_btw
svr_lin = SVR(kernel='linear', C=c_val)
svr_rbf = SVR(kernel='rbf', C=c_val, gamma=best_search_gamma_2_btw)
svr_poly = SVR(kernel='poly', C=c_val, degree=best_search_degree_2_btw)
SVR_Linear = svr_lin.fit(X_train2,Y_train2).predict(X_test2)
SVR_Rbf = svr_rbf.fit(X_train2,Y_train2).predict(X_test2)
SVR_Poly = svr_poly.fit(X_train2,Y_train2).predict(X_test2)
plt.scatter(np.arange(len(Y_train1)),Y_train2, edgecolors='b',alpha=0.75,s=5)
plt.scatter(np.arange(len(Y_test1)),Y_test2, edgecolors='r',alpha=0.75,s=1)
plt.scatter(np.arange(len(LM_pred)),LM_pred, edgecolors='pink',alpha=0.75,s=1)
plt.scatter(np.arange(len(SVR_Linear)),SVR_Linear,edgecolors='green',alpha=0.75,s=1)
plt.scatter(np.arange(len(SVR_Poly)),SVR_Poly, edgecolors='m',alpha=0.75,s=1)
plt.scatter(np.arange(len(SVR_Rbf)),SVR_Rbf, edgecolors='orange',alpha=0.75,s=1)
plt.show()

# option 1
for EST in regression:
    model = regression[EST]
    if (EST == 'LR'):
        params = params_LR
    else:
        params = params_SVR
    grid_rand = RandomizedSearchCV(estimator=model, n_jobs = 1,verbose = 10,cv = 3,scoring = 'neg_mean_squared_error',param_distributions = params)
    grid_rand_result = grid_rand.fit(X_train1, Y_train1)

# Show Best Parameters for both models
print('Best params: ',grid_rand_result.best_params_)
print('Best score: ', grid_rand_result.best_score_)

# Show Score for each parameter combination for both model
means = grid_rand_result.cv_results_['mean_test_score']
stds = grid_rand_result.cv_results_['std_test_score']
params = grid_rand_result.cv_results_['params']
bestmodelbtw = SVR(kernel = grid_rand_result.best_params_['kernel'], C = grid_rand_result.best_params_['C'], degree= grid_rand_result.best_params_['degree'], epsilon = grid_rand_result.best_params_['epsilon'], gamma = grid_rand_result.best_params_['gamma'])
bestmodelbtw_pred = bestmodelbtw.fit(X_train1,Y_train1).predict(X_test1)

bar_mean1_rand_linear = []
bar_stds1_rand_linear = []
bar_mean1_rand_poly = []
bar_stds1_rand_poly = []
bar_mean1_rand_rbf = []
bar_stds1_rand_rbf = []
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
    if param['kernel'] == 'linear':
        bar_mean1_rand_linear.append(mean)
        bar_stds1_rand_linear.append(stdev)
    elif param['kernel'] == 'poly': 
        bar_mean1_rand_poly.append(mean)
        bar_stds1_rand_poly.append(stdev)
    else:
        bar_mean1_rand_rbf.append(mean)
        bar_stds1_rand_rbf.append(stdev)
#linear
x = np.arange(len(bar_mean1_rand_linear))
print(x)
w = 0.5
fig, ax = plt.subplots()
fig = plt.title('Option #1 Random linear')
rect1 = plt.bar(x-w/2,bar_mean1_rand_linear,w,color = 'b')
rect2 = plt.bar(x+w/2,bar_stds1_rand_linear,w,color = 'r')
plt.show()
#poly
x = np.arange(len(bar_mean1_rand_poly))
print(x)
w = 0.5
fig, ax = plt.subplots()
fig = plt.title('Option #1 Random poly')
rect1 = plt.bar(x-w/2,bar_mean1_rand_poly,w,color = 'b')
rect2 = plt.bar(x+w/2,bar_stds1_rand_poly,w,color = 'r')
plt.show()
#rbf
x = np.arange(len(bar_mean1_rand_rbf))
print(x)
w = 0.5
fig, ax = plt.subplots()
fig = plt.title('Option #1 Random rbf')
rect1 = plt.bar(x-w/2,bar_mean1_rand_rbf,w,color = 'b')
rect2 = plt.bar(x+w/2,bar_stds1_rand_rbf,w,color = 'r')
plt.show()
# scatter รอถามอาจารย์

#scatter between y_test vs y_predict
plt.scatter(np.arange(len(Y_test1)),Y_test1, edgecolors='r',alpha=0.75,s=1)
plt.scatter(np.arange(len(bestmodelbtw_pred)),bestmodelbtw_pred, edgecolors='b',alpha=0.75,s=1)
plt.show()

# option 2
for EST in regression:
    model = regression[EST]
    if (EST == 'LR'):
        params = params_LR
    else:
        params = params_SVR
    grid_rand = RandomizedSearchCV(estimator=model, n_jobs = 1,verbose = 10,cv = 3,scoring = 'neg_mean_squared_error',param_distributions = params)
    grid_rand_result = grid_rand.fit(X_train2, Y_train2)

# Show Best Parameters for both models
print('Best params: ',grid_rand_result.best_params_)
print('Best score: ', grid_rand_result.best_score_)

# Show Score for each parameter combination for both model
means = grid_rand_result.cv_results_['mean_test_score']
stds = grid_rand_result.cv_results_['std_test_score']
params = grid_rand_result.cv_results_['params']
bestmodelbtw = SVR(kernel = grid_rand_result.best_params_['kernel'], C = grid_rand_result.best_params_['C'], degree= grid_rand_result.best_params_['degree'], epsilon = grid_rand_result.best_params_['epsilon'], gamma = grid_rand_result.best_params_['gamma'])
bestmodelbtw_pred = bestmodelbtw.fit(X_train2,Y_train2).predict(X_test2)


bar_mean2_rand_linear = []
bar_stds2_rand_linear = []
bar_mean2_rand_poly = []
bar_stds2_rand_poly = []
bar_mean2_rand_rbf = []
bar_stds2_rand_rbf = []
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
    if param['kernel'] == 'linear':
        bar_mean2_rand_linear.append(mean)
        bar_stds2_rand_linear.append(stdev)
    elif param['kernel'] == 'poly': 
        bar_mean2_rand_poly.append(mean)
        bar_stds2_rand_poly.append(stdev)
    else:
        bar_mean2_rand_rbf.append(mean)
        bar_stds2_rand_rbf.append(stdev)
#linear
x = np.arange(len(bar_mean2_rand_linear))
print(x)
w = 0.5
fig, ax = plt.subplots()
fig = plt.title('Option #2 Random linear')
rect1 = plt.bar(x-w/2,bar_mean2_rand_linear,w,color = 'b')
rect2 = plt.bar(x+w/2,bar_stds2_rand_linear,w,color = 'r')
plt.show()
#poly
x = np.arange(len(bar_mean2_rand_poly))
print(x)
w = 0.5
fig, ax = plt.subplots()
fig = plt.title('Option #2 Random poly')
rect1 = plt.bar(x-w/2,bar_mean2_rand_poly,w,color = 'b')
rect2 = plt.bar(x+w/2,bar_stds2_rand_poly,w,color = 'r')
plt.show()
#rbf
x = np.arange(len(bar_mean2_rand_rbf))
print(x)
w = 0.5
fig, ax = plt.subplots()
fig = plt.title('Option #2 Random rbf')
rect1 = plt.bar(x-w/2,bar_mean2_rand_rbf,w,color = 'b')
rect2 = plt.bar(x+w/2,bar_stds2_rand_rbf,w,color = 'r')
plt.show()
# scatter รอถามอาจารย์

#scatter between y_test vs y_predict
plt.scatter(np.arange(len(Y_test2)),Y_test2, edgecolors='r',alpha=0.75,s=5)
plt.scatter(np.arange(len(bestmodelbtw_pred)),bestmodelbtw_pred, edgecolors='b',alpha=0.75,s=5)
plt.show()