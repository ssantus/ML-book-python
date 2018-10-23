from sklearn import datasets
import numpy as np
iris = datasets.load_iris()
X = iris.data[:, [2, 3]]
y= iris.target

#spliting the dataset
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

#Feature scaling: standarization
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit(X_train) #estimates the parameters: mean and standard deviation for EACH FEATURE DIMENSION from the training data
X_train_std=sc.transform(X_train)
X_test_std=sc.transform(X_test)

#Perceptron
from sklearn.linear_model import Perceptron
ppn = Perceptron(n_iter = 40, eta0 = 0.1, random_state = 0) #random_state=0 gives the system...
#... reproducibility of the initial shuffling of the training dataset after each epoch
ppn.fit(X_train_std, y_train) #initializing the model
y_pred = ppn.predict(X_test_std)
n_y_pred = y_pred.size
print(n_y_pred)

n_misclassification = (y_test != y_pred).sum()
print('Misclassified samples: %d' % n_misclassification)

#from the previous, the missclassification error can be calculated as:
misclassification_error = n_misclassification / n_y_pred

print('Misclassificication error: %.2f' % misclassification_error)

#Therefore the accuracy in two forms:
print('Accuracy: %.2f' % (1 - misclassification_error))

from sklearn.metrics import accuracy_score
print('Accuracy sci-kit: %.2f' % accuracy_score(y_test, y_pred))

from Plot_decision_regions_function_contoured import plot_decision_regions_contoured
import matplotlib.pyplot as plt

X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))
plot_decision_regions_contoured(X=X_combined_std, y=y_combined, classifier=ppn, test_idx=range(105,150))
plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.legend(loc='upper left')
plt.show()
