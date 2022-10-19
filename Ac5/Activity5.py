from pydoc import describe
from statistics import mean
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn import model_selection
from sklearn import preprocessing
from sklearn import metrics
import pandas_datareader.data as web
from scipy.stats import zscore

#Read Stock data form web
stk_tickers = ['MSFT', 'IBM', 'GOOGL']
ccy_tickers = ['DEXJPUS', 'DEXUSUK']
idx_tickers = ['SP500', 'DJIA', 'VIXCLS']
stk_data = web.DataReader(stk_tickers, 'yahoo')
ccy_data = web.DataReader(ccy_tickers, 'fred')
idx_data = web.DataReader(idx_tickers, 'fred')

ccy_data = ccy_data.dropna(thresh=2)
idx_data = idx_data.dropna(thresh=2)
# print(stk_data.describe())
# print(ccy_data.describe())
# print(idx_data.describe())
ccy_data=ccy_data.fillna(ccy_data.mean())
idx_data["DJIA"] = idx_data["DJIA"].fillna(idx_data["DJIA"].mode())
idx_data.replace({object:np.nan},regex=True,inplace=True)
idx_data = idx_data.fillna(idx_data.median())
print(stk_data.info())
print(ccy_data.info())
print(idx_data.info())

#Select column
base = stk_data.loc[:, ('Adj Close', 'MSFT')]
X1 = stk_data.loc[:, ('Adj Close', ('GOOGL', 'IBM'))]
X2 = ccy_data
X3 = idx_data
X3.to_csv("X3.csv")
#Standardize
X1 = X1.apply(zscore)
X2 = X2.apply(zscore)
X3 = X3.apply(zscore)
return_period = 3
Y = base.shift(-return_period)
X4_3DT = base.diff(3*return_period).shift(-3*return_period)
X4_6DT = base.diff(6*return_period).shift(-6*return_period)
X4_12DT = base.diff(12*return_period).shift(-12*return_period)
X4 = pd.concat([X4_3DT, X4_6DT, X4_12DT], axis=1)
X4.columns = ['MSFT_3DT', 'MSFT_6DT', 'MSFT_12DT']
#ถ้าไม่dropnaข้อมูลทั้งแถวจะเป็นna
X4.dropna(inplace=True)
X4 = X4.apply(zscore)
#forming dataset
X = pd.concat([X1, X2, X3, X4], axis=1)
dataset = pd.concat([Y, X], axis=1)
#drop na
dataset.dropna(inplace=True)
print(dataset)
print(dataset.info()) 
# Assign X, Y (drop datetime index)
Y = dataset.filter([dataset.columns[0]])
X = dataset.drop(columns = [dataset.columns[0]])
Y=Y.reset_index(drop = True)
X=X.reset_index(drop = True)
print(Y)
print(X)
# feature selection (correlation)
corr = X.corr()
lower = pd.DataFrame(np.tril(corr, -1),columns = corr.columns)
# drop columns if corr value > 0.9
to_drop = [column for column in lower if any(lower[column] > 0.9)]
X.drop(to_drop, inplace=True, axis=1)
print(X)
# Train / Test Preparation
train_size = int(np.floor(0.7 * len( X )))
X_train, X_test = X[0:train_size], X[train_size:len(X)]
Y_train, Y_test = Y[0:train_size], Y[train_size:len(X)]
# Cross Validation Model
# set k-fold crossvalidation with shuffle
num_fold = 4
seed = 42
kfold = model_selection.KFold(n_splits=num_fold, shuffle = True, random_state=seed)
# Model selection
model_LM = LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=False)
#c_val ลองอย่างน้อย 3 ค่า [0.1, 1, 10, 100]
dud = [ 1, 10, 100]
AVG_Linear_Model= [ 1, 10, 100]
AVG_SVR_linear  =[ 1, 10, 100]
AVG_SVR_rbf  =[ 1, 10, 100]
AVG_SVC_poly =[ 1, 10, 100]
for i in range(len(dud)):
    c_val = dud[i]
    svr_lin = SVR(kernel='linear', C=c_val)
    svr_rbf = SVR(kernel='rbf', C=c_val, gamma=1)
    svr_poly = SVR(kernel='poly', C=c_val, degree=3)

    # Calculate accuracy score for each model

    score_LM = model_selection.cross_val_score(model_LM, X_train, Y_train, cv=kfold)
    score_lin = model_selection.cross_val_score(svr_lin, X_train, Y_train, cv=kfold)
    score_rbf = model_selection.cross_val_score(svr_rbf, X_train, Y_train, cv=kfold)
    score_poly = model_selection.cross_val_score(svr_poly, X_train, Y_train, cv=kfold)

    # View score k-fold
    # # Valication score comparison
    score = pd.DataFrame({'Linear Model':score_LM,'SVR_linear':score_lin, 'SVR_rbf': score_rbf, 'SVR_poly':
    score_poly})
    score_mean = pd.DataFrame({'AVG Linear Model':[score_LM.mean()],'AVG SVR_linear':[score_lin.mean()],
    'AVG SVR_rbf': [score_rbf.mean()], 'AVG SVC_poly': [score_poly.mean()]})
    print(score)
    print(score_mean)
    AVG_Linear_Model[i] = score_LM.mean()
    AVG_SVR_linear[i]  = score_lin.mean()
    AVG_SVR_rbf[i]  = score_rbf.mean()
    AVG_SVC_poly[i] =score_poly.mean()
print(AVG_Linear_Model)
plt.plot(AVG_Linear_Model, color = 'r')
plt.plot(AVG_SVR_linear, color = 'g')
plt.plot(AVG_SVR_rbf, color = 'b')
plt.plot(AVG_SVC_poly, color = 'y')
plt.show()

c_val = 10
svr_lin = SVR(kernel='linear', C=c_val)
svr_rbf = SVR(kernel='rbf', C=c_val, gamma=0.01)
svr_poly = SVR(kernel='poly', C=c_val, degree=2)
LM_pred = model_LM.fit(X_train, Y_train).predict(X_test)
svr_lin_pred = svr_lin.fit(X_train,Y_train).predict(X_test)
svr_rbf_pred = svr_rbf.fit(X_train,Y_train).predict(X_test)
svr_poly_pred = svr_poly.fit(X_train,Y_train).predict(X_test)
X_test.reset_index(inplace = True)
X_test.drop(columns = [X_test.columns[0]],inplace = True)
print(len(LM_pred))
print(X_test)
# plt.scatter(X_test["DEXJPUS"],LM_pred, c='magenta')
# plt.show()
LM_MSE = metrics.mean_squared_error(Y_test, LM_pred)
LM_r2 = metrics.r2_score(Y_test, LM_pred)
print (LM_MSE)
print(LM_r2)

svr_lin_MSE = metrics.mean_squared_error(Y_test, svr_lin_pred)
svr_lin_r2 = metrics.r2_score(Y_test, svr_lin_pred)
print (svr_lin_MSE)
print(svr_lin_r2)


svr_rbf_MSE = metrics.mean_squared_error(Y_test, svr_rbf_pred)
svr_rbf_r2 = metrics.r2_score(Y_test, svr_rbf_pred)
print (svr_rbf_MSE)
print(svr_rbf_r2)

svr_poly_MSE = metrics.mean_squared_error(Y_test, svr_poly_pred)
svr_poly_r2 = metrics.r2_score(Y_test, svr_poly_pred)
print (svr_poly_MSE)
print(svr_poly_r2)

# plt.bar(["LM_MSE","svr_lin_MSE","svr_rbf_MSE","svr_poly_MSE"],[LM_MSE,svr_lin_MSE,svr_rbf_MSE,svr_poly_MSE])
plt.bar(["LM_r2","svr_lin_r2","svr_rbf_r2","svr_poly_r2"],[LM_r2,svr_lin_r2,svr_rbf_r2,svr_poly_r2])
plt.show()