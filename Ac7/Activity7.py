from pydoc import describe
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn import model_selection
from sklearn import tree
#exoplo
df=pd.read_csv("Coffee-modified.csv")
df=df.filter(['Total.Cup.Points', 'Species','Country.of.Origin','Processing.Method', 'Aroma', 'Flavor', 'Aftertaste','Acidity','Body', 'Balance', 'Uniformity', 'Moisture', 'altitude_mean_meters'])
df.dropna(inplace=True)

#assign X Y
Y = df.filter([df.columns[0]])
X = df.drop(columns = [df.columns[0]])
Y=Y.reset_index(drop = True)
X=X.reset_index(drop = True)
print(X)
print(Y)

#process Y from value to Coffee bean grade
# define Bean_Grade = [1,2,3] using

rating_pctile = np.percentile( Y, [75, 90])
grade1 = 0
grade2 = 0
grade3 = 0
bg = []
for i in range(len(Y.index)):
    if (Y["Total.Cup.Points"][i] < rating_pctile [0]): 
        grade1=grade1+1
        bg.append(1)
    if (rating_pctile [0] <= Y["Total.Cup.Points"][i] < rating_pctile [1]): 
        grade2=grade2+1
        bg.append(2) 
    if (Y["Total.Cup.Points"][i] >= rating_pctile[1]): 
        grade3=grade3+1
        bg.append(3)
Y["Bean_grade"] = bg
Y.drop(["Total.Cup.Points"],axis =1,inplace= True)
print(Y)
figdata = pd.DataFrame([[1,grade1],[2,grade2],[3,grade3]],columns = ["Bean_grade","NSamples"])
# Visualize Bar Graph of Number of Samples for each Bean Grade
# ตัวอย่างการลองใช้ plotly express library
fig = px.bar( figdata, x = 'Bean_grade', y = 'NSamples', color='NSamples', range_y=[0.0,1000])
fig.show()
# 7.1(c)
standard_scaler = preprocessing.StandardScaler()
print(X.info())
Xnum = X.filter(["Aroma","Flavor","Aftertaste","Acidity","Body","Balance","Uniformity","Moisture","altitude_mean_meters"])
Xchar = X.filter(["Species","Country.of.Origin","Processing.Method"])
print(Xnum)
print(Xchar)
Xnum = pd.DataFrame(standard_scaler.fit_transform(Xnum.values),index = Xnum.index,columns=Xnum.columns)
onehot = pd.get_dummies(Xchar, columns = ["Species","Country.of.Origin","Processing.Method"], drop_first=True )
print(onehot)
print(Xnum)
X = onehot.join(Xnum)
corr = X.corr()
lower = pd.DataFrame(np.tril(corr, -1),columns = corr.columns)
to_drop = [column for column in lower if any(lower[column] > 0.8)]
X.drop(to_drop, inplace=True, axis=1)
print(X.info())
# option 1
# test_size = int(np.floor(0.3 * len( X )))
# train_size = int(np.floor(0.7 * len( X )))
# X_train, X_test = X[0:train_size], X[train_size:len(X)]
# Y_train, Y_test = Y[0:train_size], Y[train_size:len(X)]
# option 2
rseed=42
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=0.3, random_state=rseed)
# 7.2(a)
k = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 25, 35]
KNN_Score = []
for i in range(len(k)):
    # Model Training
    modelKNN = KNeighborsClassifier(n_neighbors=k[i], p=2)
    modelKNN.fit(X_train,Y_train)
    # Model Testing
    Y_pred= modelKNN.predict(X_test)
    KNNScore = accuracy_score(Y_test, Y_pred)
    print(KNNScore)
    KNN_Score.append(KNNScore)
data = {"k" : k,"KNN_Score" : KNN_Score}
figshow = pd.DataFrame(data)
fig = px.bar( figshow, x = 'k', y = 'KNN_Score', color='KNN_Score', range_y=[0.7,1.0])
fig.show()

# Print Confusion Matrix and Classification Report for best k
# option 1 k=3,5 (index = 1,2)
# option 2 k=11 (index = 5)
# Model Training
modelKNN = KNeighborsClassifier(n_neighbors=k[5], p=2)
modelKNN.fit(X_train,Y_train)
# Model Testing
Y_pred= modelKNN.predict(X_test)
KNNScore = accuracy_score(Y_test, Y_pred)

print(KNNScore)
print('Confusion Matrix: ')
print(confusion_matrix(Y_test, Y_pred))
print('Classification Report: ')
print(classification_report(Y_test, Y_pred))
# 7.2(B)
# Decision Tree parameter
ASM_function = ["entropy", "gini"]
maxD = [4, 5, 6, None] # try at least 2 values
# Model Training

for i in ASM_function:
    for j in maxD:
        ModelDT = DecisionTreeClassifier(criterion=i, splitter='best',max_depth = j )
        ModelDT.fit(X_train,Y_train)
        # Model Testing
        Y_pred= ModelDT.predict(X_test)
        DTScore = accuracy_score(Y_test, Y_pred)
        print("criterion = ",i,"max_depth =",j)
        print(DTScore)
# Print Confusion Matrix and Classification Report for best k
# option 1 criterion =  entropy max_depth = 4
# option 2 criterion =  entropy max_depth = 6 , criterion =  gini max_depth = 5
ModelDT = DecisionTreeClassifier(criterion="entropy", splitter='best',max_depth = 6 )
ModelDT.fit(X_train,Y_train)
# Model Testing
Y_pred= ModelDT.predict(X_test)
DTScore = accuracy_score(Y_test, Y_pred)
print("criterion = ",i,"max_depth =",j)
print(DTScore)
print('Confusion Matrix: ')
print(confusion_matrix(Y_test, Y_pred))
print('Classification Report: ')
print(classification_report(Y_test, Y_pred))
# Visualize Decision Tree

feature_names = X_train.columns
Labels = str(np.unique(Y_train))
print(Labels)
tree.plot_tree( ModelDT,feature_names = feature_names,class_names = Labels,rounded = True,filled = True, fontsize=9)
plt.show()

# 7.2(C)
# Random Forest parameter
ASM_function = ['entropy', 'gini']
nEstimator = 100
nJob = 2
rState = 10
for i in ASM_function :
    # Model Training
    RandomF = RandomForestClassifier(criterion=i,n_estimators=nEstimator, n_jobs=nJob, random_state=rState)
    RandomF.fit(X_train,Y_train)
    # Model Testing
    Y_pred= RandomF.predict(X_test)
    RFScore = accuracy_score(Y_test, Y_pred)
    print("criterion = ",i)
    print(RFScore)
# option 1 criterion = เท่ากัน
# option 2 criterion = entropy
# Print Confusion Matrix and Classification Report for best k
# Model Training
RandomF = RandomForestClassifier(criterion="entropy",n_estimators=nEstimator, n_jobs=nJob, random_state=rState)
RandomF.fit(X_train,Y_train)
# Model Testing
Y_pred= RandomF.predict(X_test)
RFScore = accuracy_score(Y_test, Y_pred)
print(RFScore)
print('Confusion Matrix: ')
print(confusion_matrix(Y_test, Y_pred))
print('Classification Report: ')
print(classification_report(Y_test, Y_pred))
# Visualize Feature Important Score
feature_imp = pd.Series(RandomF.feature_importances_, index = feature_names).sort_values(ascending=False)

# Creating a bar plot
plt.figure(figsize=(15,15))
sns.barplot(x=feature_imp, y=feature_imp.index)
plt.show()

# Visualize selected estimator [0-5] tree structure of Random forest
fig, axes = plt.subplots(nrows = 1,ncols = 5,figsize = (10,2), dpi=150)
for index in range(0, 5):
    tree.plot_tree( RandomF.estimators_[index],feature_names = feature_names,class_names= Labels,filled = True,ax = axes[index])
    axes[index].set_title('Estimator: ' + str(index), fontsize = 11)
plt.show()

# Create Model List
classification = { "KNN": KNeighborsClassifier(), "DT": DecisionTreeClassifier(), "RF": RandomForestClassifier() }
# Create Parameter Dictionary for KNN
K_list = [1, 3, 5, 7, 9 , 11, 13, 15, 17, 19, 21, 23, 25, 35, 45]
KNN_param = dict(n_neighbors=K_list)

# Create Parameter Dictionary for Decision Tree
ASM_function = ["entropy", "gini"]
maxD = [ 4, 5, 6, None]
maxF = ["auto", "log2", None]
minSample = [1,2, 4]
DT_param= dict(criterion=ASM_function, max_depth = maxD, min_samples_leaf = minSample, max_features = maxF)
# Create Parameter Dictionary for Random Forest (including same parameters as Decision Tree)
nEst = [10, 30, 50, 100]
RF_param = dict(n_estimators = nEst, criterion=ASM_function, max_depth = maxD, min_samples_leaf = minSample,max_features = maxF)
# Perform GridsearchCV() for each classification model
for EST in classification:
    model = classification[EST]
    if (EST == 'KNN'):
        params = KNN_param
    elif(EST == 'DT'):
        params = DT_param
    else:
        params = RF_param
    grid = GridSearchCV( estimator = model,n_jobs = 1,verbose = 10,scoring = 'accuracy', cv = 2,param_grid = params )
grid_result = grid.fit(X_train,Y_train)
print('Best params: ',grid_result.best_params_)
print('Best score: ', grid_result.best_score_)
mean = grid_result.cv_results_['mean_test_score']
std = grid_result.cv_results_['std_test_score']
param = grid_result.cv_results_['params']
bar_mean_entropy = []
bar_mean_gini = []
bar_std_entropy = []
bar_std_gini = []
bar_param_entropy = []
bar_param_gini = []

for mean, stdev, param in zip(mean, std, param):
    print("%f (%f) with: %r" % (mean, stdev, param))
    if param['criterion'] == 'entropy':
        bar_mean_entropy.append(mean)
        bar_std_entropy.append(stdev)
        bar_param_entropy.append("md : "+ str(param['max_depth']) + ", mf : " + str(param['max_features']) + ", msl : " + str(param['min_samples_leaf']) + ", ne : " + str(param['n_estimators']))
    else:
        bar_mean_gini.append(mean)
        bar_std_gini.append(stdev)
        bar_param_gini.append("md : "+ str(param['max_depth']) + ", mf : " + str(param['max_features']) + ", msl : " + str(param['min_samples_leaf']) + ", ne : " + str(param['n_estimators']))



x = np.arange(len(bar_mean_entropy))
w = 0.5
fig, ax = plt.subplots()
fig = plt.title('Criterion : entropy')
rect1 = plt.bar(x-w/2,bar_mean_entropy,w,color = 'b')
rect2 = plt.bar(x+w/2,bar_std_entropy,w,color = 'r')
ax.set_xticks(x, labels= bar_param_entropy,fontsize=6,rotation = 90)
plt.subplots_adjust(bottom=0.20)
plt.show()

fig, ax = plt.subplots()
fig = plt.title('Criterion : gini')
rect1 = plt.bar(x-w/2,bar_mean_gini,w,color = 'b')
rect2 = plt.bar(x+w/2,bar_std_gini,w,color = 'r')
ax.set_xticks(x, labels= bar_param_gini,fontsize=6,rotation = 90)
plt.subplots_adjust(bottom=0.20)
plt.show()