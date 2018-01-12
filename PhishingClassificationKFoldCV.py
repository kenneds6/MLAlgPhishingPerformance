from sklearn import svm, metrics
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

import time
import arff, numpy as np
%matplotlib inline
import matplotlib.pyplot as plt

#Keep track of elapsed time
start_time = time.time()

#Load and handle data
dataset = arff.load(open('Training Dataset.arff'))
data = np.array(dataset['data']).astype(np.float)

X = data[:,0:29]
Y = data[:,29]


#GNB Classifier
gnb = GaussianNB()
gnbScores = cross_val_score(gnb, X, Y, cv = 10)

#SVM Classification
svmClassifier = svm.SVC()
svmScores = cross_val_score(svmClassifier, X, Y, cv = 10)

#Random Forest Classifier
rfClassifier = RandomForestClassifier(n_estimators=10)
rfScores = cross_val_score(rfClassifier, X, Y, cv = 10)


#Perceptron Backpropagation Classifier
mlpClassifier = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
mlpScores = cross_val_score(mlpClassifier, X, Y, cv = 10)


fig = plt.figure() 
plt.plot(gnbScores, label = 'Naives Bayes')
plt.plot(svmScores, label = 'SVM')
plt.plot(rfScores, label = 'Random Forest')
plt.plot(mlpScores, label = 'Perceptron')

plt.xlabel('Bin number')
plt.ylabel('Accuracy')

plt.title("Comparison of Accuracy of 4 different Classifiers using K-Fold Cross Validation")

plt.legend()

plt.show()


elapsed_time = time.time() - start_time
print("Elapsed Time: ", elapsed_time, '\n')

print("GNB Classifier:\n","Mean: ", np.mean(gnbScores), '\n', "Standard Deviation: ", np.std(gnbScores))
print("SVM Classifier:\n","Mean: ", np.mean(svmScores), '\n', "Standard Deviation: ", np.std(svmScores))
print("Random Forest Classifier:\n","Mean: ", np.mean(rfScores), '\n', "Standard Deviation: ", np.std(rfScores))
print("Multi-Layer Perceptron Backpropagation Classifier:\n","Mean: ", np.mean(mlpScores), '\n', "Standard Deviation: ", np.std(mlpScores))
