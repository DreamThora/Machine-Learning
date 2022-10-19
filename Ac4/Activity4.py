from pydoc import describe
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from scipy.stats import zscore
df = pd.read_csv("CarPrice.csv")
# print(df)
# print(df.describe())
df.drop(['car_ID','CarName'],inplace= True,axis = 1)
df_cont = df.filter(["symboling","wheelbase","carlength","carwidth","carheight","curbweight","enginesize","boreratio","stroke","compressionratio","horsepower","peakrpm","citympg","highwaympg","price"])
df_cata = df.filter(["fueltype","aspiration","doornumber","carbody","drivewheel","enginelocation","enginetype","cylindernumber","fuelsystem"])
XOriginal = df_cont.drop(["price"],axis =1)
Y = df_cont.filter(["price"])
df_cont.drop(columns = ['price'],inplace = True,axis = 1)
# split continuos and catagories
# print (df_cont.info())
# print (df_cata.info())
#standardize
df_cont = df_cont.apply(zscore)
# calculate correlation
corr = df_cont.corr()
lower = pd.DataFrame(np.tril(corr, -1),columns = corr.columns)
# print(lower)
# drop columns if corr value > 0.86
to_drop = [column for column in lower if any(lower[column] > 0.86)]
df_cont.drop(to_drop, inplace=True, axis=1)
# print (to_drop)
# print (df_cont)
#one hot catagories
onehot = pd.get_dummies(df_cata, columns = ['fueltype','aspiration','doornumber','carbody','drivewheel','enginelocation','enginetype','cylindernumber','fuelsystem'], drop_first=True)
print(onehot)
all = df_cont.join(onehot)
print(all)
#4.2
pca = PCA()
X_pca = pca.fit_transform(all)
R1= range(len(all.columns))
# plt.bar(R1,pca.explained_variance_ratio_)
# plt.show()
print('Explained Variance ratio = ', pca.explained_variance_ratio_)
print('Explained Variance (eigenvalues) = ', pca.explained_variance_)
print('--------------------------------------------')
print('PCA components (eigenvectors) along row ')
print(pca.components_[0:10])
x_n_component = ["5","10","15","20","25","30"]
for i in range (6):
    pca2 = PCA(n_components=(i+1)*5)
    X_pca_2 = pca2.fit_transform(all)
    x_n_component[i] = X_pca_2
print(x_n_component)
print('Explained Variance ratio = ', pca2.explained_variance_ratio_)
print('Explained Variance (eigenvalues) = ', pca2.explained_variance_)
print('--------------------------------------------')
print('PCA2 components (eigenvectors) ')
print(pca2.components_[0:5])
R2= range(len(all.columns))
# plt.bar(R2,pca.explained_variance_ratio_)
# plt.show()
# 4.3
# without pca
rseed = 42
x_train_set, x_test, y_train_set, y_test = train_test_split(all, Y, test_size = 0.3, random_state = rseed)
x_train, x_validate, y_train, y_validate = train_test_split(x_train_set, y_train_set, test_size = 0.3, random_state = rseed)
# Perform Linear Regression -> All variables
lr = LinearRegression()
# train
lr.fit(x_train, y_train)
# validate
y_pred_lr = lr.predict(x_validate)
# test
y_test_pred_lr = lr.predict(x_test)
# Measure Accuracy Validation and Test
r2accvalid = [0,5,10,15,20,25,30]
r2accvalid[0] = r2_score(y_pred_lr, y_validate)
r2acctest = [0,5,10,15,20,25,30]
r2acctest[0] = r2_score(y_test_pred_lr, y_test)
#print(lr.score(x_validate, y_validate))
#print(lr.score(x_test, y_test))
msevalid = [0,5,10,15,20,25,30]
msevalid[0] = mean_squared_error(y_pred_lr, y_validate)
msetest = [0,5,10,15,20,25,30]
msetest[0] = mean_squared_error(y_test_pred_lr, y_test)
#with pca
for i in range (6):
    x_train_set, x_test, y_train_set, y_test = train_test_split(x_n_component[i], Y, test_size = 0.3, random_state = rseed)
    x_train, x_validate, y_train, y_validate = train_test_split(x_train_set, y_train_set, test_size = 0.3, random_state = rseed)
    # Perform Linear Regression -> All variables
    lr = LinearRegression()
    # train
    lr.fit(x_train, y_train)
    # validate
    y_pred_lr = lr.predict(x_validate)
    # test
    y_test_pred_lr = lr.predict(x_test)
    # Measure Accuracy Validation and Test
    r2accvalid[i+1] = r2_score(y_pred_lr, y_validate)
    r2acctest[i+1] = r2_score(y_test_pred_lr, y_test)
    #print(lr.score(x_validate, y_validate))
    #print(lr.score(x_test, y_test))
    msevalid[i+1] = mean_squared_error(y_pred_lr, y_validate)
    msetest[i+1] = mean_squared_error(y_test_pred_lr, y_test)
# plt.bar(["NO_PCA","PCA_5","PCA_10","PCA_15","PCA_20","PCA_25","PCA_30"],r2accvalid)
# plt.plot(r2acctest,color= 'r')
plt.bar(["NO_PCA","PCA_5","PCA_10","PCA_15","PCA_20","PCA_25","PCA_30"],msevalid)
plt.plot(msetest,color = 'r')

plt.show()