from sklearn import svm
from sklearn.model_selection import train_test_split
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

#Keep track of score values
gnbArr = []
svmcArr = []
rfcArr = []
mlpcArr = []

# Loop through 100 times with different random states values to 
# analyse the performance of the different algorithms.  Store in an
# array and calculate the mean and std deviation.
for i in range(0,99):

    train_set, test_set = train_test_split(data, test_size=0.2, random_state = i + 1)
    testX = test_set[:,0:29]
    testY = test_set[:,29]
    trainX = train_set[:,0:29]
    trainY = train_set[:,29] 

    #GNB Classifier
    gnb = GaussianNB()
    gnb.fit(trainX, trainY)
    gnbArr.append(gnb.score(testX, testY, sample_weight = None))
    #print("GNB Classifier",gnb.score(testX, testY, sample_weight = None), '\n')

    #SVM Classification
    svmClassifier = svm.SVC()
    svmClassifier.fit(trainX, trainY)
    svmcArr.append(svmClassifier.score(testX, testY, sample_weight = None))
    #print("SVM Classifier",svmClassifier.score(testX, testY, sample_weight = None),'\n') 

    #Random Forest Classifier
    rfClassifier = RandomForestClassifier(n_estimators=10)
    rfClassifier.fit(trainX, trainY)
    rfcArr.append(rfClassifier.score(testX, testY, sample_weight = None))
    #print("Random Forest Classifier",rfClassifier.score(testX, testY, sample_weight = None),'\n')

    #Perceptron Backpropagation Classifier
    mlpClassifier = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
    mlpClassifier.fit(trainX, trainY)
    mlpcArr.append(mlpClassifier.score(testX, testY, sample_weight = None))
    #print ("Perceptron Backpropagation Classifier", mlpClassifier.score(testX, testY, sample_weight = None),'\n')

fig = plt.figure() 
plt.plot(gnbArr, label = 'Naives Bayes')
plt.plot(svmcArr, label = 'SVM')
plt.plot(rfcArr, label = 'Random Forest')
plt.plot(mlpcArr, label = 'Perceptron')

plt.xlabel('Iteration')
plt.ylabel('Accuracy')

plt.title("Comparison of Accuracy of 4 different Classifiers using Random Subsampling")

plt.legend()

plt.show()
    
elapsed_time = time.time() - start_time
print("Elapsed Time: ", elapsed_time, '\n')

print("GNB Classifier:\n","Mean: ", np.mean(gnbArr), '\n', "Standard Deviation: ", np.std(gnbArr))
print("SVM Classifier:\n","Mean: ", np.mean(svmcArr), '\n', "Standard Deviation: ", np.std(svmcArr))
print("Random Forest Classifier:\n","Mean: ", np.mean(rfcArr), '\n', "Standard Deviation: ", np.std(rfcArr))
print("Multi-Layer Perceptron Backpropagation Classifier:\n","Mean: ", np.mean(mlpcArr), '\n', "Standard Deviation: ", np.std(mlpcArr))
