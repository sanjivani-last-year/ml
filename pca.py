import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn import datasets
%matplotlib inline
plt.style.use('seaborn')
iris = datasets.load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target, name='FlowerType')
X.head()
plt.figure(4, figsize=(8, 6))
plt.clf()

# Plot the training points
plt.scatter(X['sepal length (cm)'], X['sepal width (cm)'], s=35, c=y, cmap=plt.cm.brg)
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.title('Sepal length vs. Sepal width')
plt.show()
x = StandardScaler().fit_transform(X)

pca = PCA(n_components=2)
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data = principalComponents,columns = ['principal component 1', 'principal component 2'])
principalDf.head(5)

principalDf.shape

colors = ['r', 'g', 'b']
plt.scatter(principalComponents[:,0],principalComponents[:,1],c=y)



###################################################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
iris_data = pd.read_csv('Iris.csv')

iris_data.head(5)
iris_data.isnull().sum()
iris_data.dtypes
iris_data.shape
iris_data.describe()
x = iris_data.drop('Species', axis=1)
y = iris_data['Species']
x.head()
y.head()
#  Preprocess the dataset
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)
x_scaled
#   PCA
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
x_pca = pca.fit_transform(x_scaled)
#  Analyze the results
# Explained variance ratio
explained_variance_ratio = pca.explained_variance_ratio_
print("Explained variance ratio:", explained_variance_ratio)
plt.scatter(x_pca[:, 0], x_pca[:, 1])
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA on Iris Dataset')
plt.show()
