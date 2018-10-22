#This exercise analyzes iris dataset
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#------------
#Pre process
#------------

df = pd.read_csv('C:/Users/santus/Documents/GitHub/ML-book-python/datasets/iris.data',
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
#Process
#--------
from Perceptron import Perceptron

ppn = Perceptron (eta=0.1, n_iter= 6)
ppn.fit(X, y)

plt.figure(1)
plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Number of misclassifications')

#perceptron has converged to the solution after the 6th Epochs

#---------
#Post Process
#--------

from Plot_decision_regions_function import plot_decision_regions

plt.figure(2)
plot_decision_regions(X, y, classifier=ppn)
plt.xlabel('sepal length [cm]')
plt.ylabel('petal length [cm]')
plt.legend(loc='upper left')

plt.show()
