#This exercise analyzes iris dataset
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#------------
#Pre process
#------------

df = pd.read_csv('C:/Users/escalos/Documents/GitHub/ML-book-python/datasets/iris.data',
 header=None)
tail=df.tail() #Picks the last 5 element in the series, in this case this last 5 elements are 5 vectors (df is a matrix, so a vector of vectors EZPZ)
# print(tail)
# print(df)

#extract the first 100 class labels (50 iris-setosa, 50 iris-versicolor) and label each class as -1 and 1
y=df.iloc[0:100, 4].values #iloc function allows you to access to the element by index, loc by tag??
y= np.where(y == 'Iris-setosa', -1, 1)#numpy, allows you to search and substitute
#extract data
X=df.iloc[0:100, [0, 2]].values #X contains the sepal length-0 and petal length-2
#plot data
plt.figure(0)
plt.scatter(X[:50, 0], X[:50, 1], color = 'red', marker = 'o', label = 'setosa')
plt.scatter(X[50:100, 0], X[50:100, 1], color = 'blue', marker = 'x', label = 'versicolor')
plt.xlabel('petal length')
plt.ylabel('sepal length')
plt.legend(loc= 'upper left')

#---------
# Learning
#--------
#Perceptron
from Perceptron import Perceptron

ppn = Perceptron (eta=0.1, n_iter= 6)
ppn.fit(X, y)

plt.figure(1)
plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Number of misclassifications')

#perceptron has converged to the solution after the 6th Epochs

# Perceptron Evaluation
from Plot_decision_regions_function import plot_decision_regions

plt.figure(2)
plot_decision_regions(X, y, classifier=ppn)
plt.xlabel('sepal length [cm]')
plt.ylabel('petal length [cm]')
plt.legend(loc='upper left')


#ADALINE
from Adaline import AdalineGD

plt.figure(2)
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))
ada1 = AdalineGD(n_iter=10, eta=0.01).fit(X, y)
ax[0].plot(range(1, len(ada1.cost_) + 1), np.log10(ada1.cost_), marker='o')
ax[0].set_xlabel('Epochs')
ax[0].set_ylabel('log(Sum-squared-error)')
ax[0].set_title('Adaline - Learning rate 0.01')
ada2 = AdalineGD(n_iter=10, eta=0.0001).fit(X, y)
ax[1].plot(range(1, len(ada2.cost_) + 1), ada2.cost_, marker='o')
ax[1].set_xlabel('Epochs')
ax[1].set_ylabel('Sum-squared-error')
ax[1].set_title('Adaline - Learning rate 0.0001')
plt.show()
#Adaline Evaluation
#ADALINE after standarization of the data
# feature scaling:standarization

X_std = np.copy(X)
X_std[:,0] = (X[:,0] - X[:,0].mean()) / X[:,0].std()
X_std[:,1] = (X[:,1] - X[:,1].mean()) / X[:,1].std()

ada = AdalineGD(n_iter=15, eta=0.01)
plt.figure(3)
ada.fit(X_std, y)
plot_decision_regions(X_std, y, classifier=ada)
plt.title('Adaline - Gradient Descent')
plt.xlabel('sepal length [standardized]')
plt.ylabel('petal length [standardized]')
plt.legend(loc='upper left')
plt.figure(4)
plt.plot(range(1, len(ada.cost_) + 1), ada.cost_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Sum-squared-error')
plt.show()
