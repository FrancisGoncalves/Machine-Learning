# ------------------------------------------- WSU - CPT_S 570 - HW2 -----------------------------------------
# -------------------------------------------- Polynomial Kernel --------------------------------------------
# ---------------------------------------- Author: Francisco Goncalves ------------------------------------------

import numpy as np
from loadingData import reducedTrainingData, testingData, validationData
from sklearn.svm import SVC


def polynomialKernel():
    print('This will take a while. Please wait...')
    accuracy_testing = np.zeros(3)
    accuracy_training = np.zeros(3)
    accuracy_validation = np.zeros(3)
    numberSV = np.zeros((3, 10))
    for d in range(2, 5):
        print('Computing for polynomial of degree' + str(d))
        clf = SVC(kernel='poly', degree=d, C=0.1, max_iter=1000)
        clf.fit(reducedTrainingData.examples, reducedTrainingData.labels)
        numberSV[d-2, :] = clf.n_support_
        accuracy_training[d-2] = clf.score(reducedTrainingData.examples, reducedTrainingData.labels)
        accuracy_validation[d-2] = clf.score(validationData.examples, validationData.labels)
        accuracy_testing[d-2] = clf.score(testingData.examples, testingData.labels)

        for p in range(3):
            print('The number of support vectors for each class for polynomial of degree ' + str(p+2) + 'is ')
            print(numberSV[p, :])
            print('The training accuracy for polynomial of degree' + str(p+2) + 'is ' + str(accuracy_training[p]))
            print('The validation accuracy for polynomial of degree' + str(p+2) + 'is ' + str(accuracy_validation[p]))
            print('The testing accuracy for polynomial of degree' + str(p+2) + 'is ' + str(accuracy_testing[p]))



