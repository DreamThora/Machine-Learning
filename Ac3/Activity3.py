import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from scipy.stats import mode
from scipy.stats import zscore
from IPython.display import Image
df_wine = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data', header=None)
df_wine.columns = ['Class label', 'Alcohol', 'Malic acid', 'Ash',
 'Alcalinity of ash', 'Magnesium', 'Total phenols',
 'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins',
 'Color intensity', 'Hue',
 'OD280/OD315 of diluted wines', 'Proline']
 

#fillna if data was int fill with mode, if data was float fill with mean
df_wine["Class label"] = df_wine["Class label"].fillna(df_wine["Class label"].mode) 
df_wine["Alcohol"] = df_wine["Alcohol"].fillna(df_wine["Alcohol"].mean) 
df_wine["Malic acid"] = df_wine["Malic acid"].fillna(df_wine["Malic acid"].mean)
df_wine["Ash"] = df_wine["Ash"].fillna(df_wine["Ash"].mean)

df_wine["Alcalinity of ash"] = df_wine["Alcalinity of ash"].fillna(df_wine["Alcalinity of ash"].mean)
df_wine["Magnesium"] = df_wine["Magnesium"].fillna(df_wine["Magnesium"].mean)
df_wine["Total phenols"] = df_wine["Total phenols"].fillna(df_wine["Total phenols"].mean)

df_wine["Flavanoids"] = df_wine["Flavanoids"].fillna(df_wine["Flavanoids"].mean)
df_wine["Nonflavanoid phenols"] = df_wine["Nonflavanoid phenols"].fillna(df_wine["Nonflavanoid phenols"].mean)
df_wine["Proanthocyanins"] = df_wine["Proanthocyanins"].fillna(df_wine["Proanthocyanins"].mean)

df_wine["Color intensity"] = df_wine["Color intensity"].fillna(df_wine["Color intensity"].mean)
df_wine["Hue"] = df_wine["Hue"].fillna(df_wine["Hue"].mean)

df_wine["OD280/OD315 of diluted wines"] = df_wine["OD280/OD315 of diluted wines"].fillna(df_wine["OD280/OD315 of diluted wines"].mean)
df_wine["Proline"] = df_wine["Proline"].fillna(df_wine["Proline"].mean)

print(df_wine)
print(df_wine.describe())
X = df_wine.copy()
XOriginal = df_wine.copy()
X.drop(["Class label"],inplace=True,axis=1)
XOriginal.drop(["Class label"],inplace=True,axis=1)
Y = df_wine.copy()
Y=Y.filter(["Class label"])
print(X)
print(Y)
X=X.apply(zscore)
print(X)
df_wine = Y.join(X)
# print(df_wine)
# sns_plot = sns.pairplot(df_wine, hue='Class label',size=2.5);
# sns_plot.savefig("pairplot.png")

pca = PCA()
X_pca = pca.fit_transform(X)
print('Explained Variance ratio = ', pca.explained_variance_ratio_)
#plt.bar(range(1,14),pca.explained_variance_ratio_)
#plt.show()
print('Explained Variance (eigenvalues) = ', pca.explained_variance_)
print('--------------------------------------------')
print('PCA components (eigenvectors) ')
print(pca.components_[0:3,:])
# PCA all variables (after standardized data)
pca2 = PCA(n_components=2)
X_PCA_2 = pca2.fit_transform(X)
# plt.bar(range(1,3),pca2.explained_variance_ratio_)
# plt.show()
print('Explained Variance (eigenvalues) = ', pca2.explained_variance_)
print('--------------------------------------------')
print('PCA2 components (eigenvectors) ')
print(pca2.components_[0:2,:])
# 3.3
n_clusters=np.unique(Y)
kmeans = KMeans(n_clusters= n_clusters.size, random_state=0)
clusters = kmeans.fit_predict(XOriginal)
# plt.scatter(XOriginal["Alcohol"], XOriginal["Malic acid"], c=clusters, edgecolors='m',alpha=0.75,s=150)
# centers = kmeans.cluster_centers_
# plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.5);
# plt.show()
# labels = np.zeros_like(clusters)
# print(clusters)
# for i in range(10):
#     mask = (clusters == i)
#     labels[mask] = mode(Y[mask])[0]
# print(accuracy_score(labels, Y))

# pca
kmeans_PCA = KMeans(n_clusters= n_clusters.size, random_state=0)
clusters_PCA = kmeans_PCA.fit_predict(X_PCA_2)
plt.scatter(X_PCA_2[:, 0], X_PCA_2[:, 1], c=clusters_PCA, edgecolors='m',alpha=0.75,s=150)
centers_pca = kmeans_PCA.cluster_centers_
plt.scatter(centers_pca[:, 0], centers_pca[:, 1], c='red', s=200, alpha=0.5);
plt.show()
