# ------------------------------------------- WSU - CPT_S 570 - HW2 ---------------------------------------------
# ----------------------------------------------- Linear Kernel -------------------------------------------------
# ---------------------------------------- Author: Francisco Goncalves ------------------------------------------

import numpy as np
from loadingData import reducedTrainingData, testingData, validationData
from sklearn.svm import SVC
import matplotlib.pyplot as plt


def linearKernel():
    print('This will take a while. Please wait...')
    # Create list with different C parameters
    C = np.zeros(9)
    n = 0
    for exp in range(-4, 5):
        # print(exp)
        C[n] = 10 ** exp
        n += 1

    # Prealocating accuracy values
    accuracy_testing = np.zeros(9)
    accuracy_training = np.zeros(9)
    accuracy_validation = np.zeros(9)
    numberSV = np.zeros(9)

    # Kernel algorithm for all C parameters
    for i in range(9):
        # print('C = ' + str(C[i]))
        clf = SVC(C=C[i], kernel='linear', max_iter=1000)
        clf.fit(reducedTrainingData.examples, reducedTrainingData.labels)
        numberSV[i] = sum(clf.n_support_)
        # Computing accuracies
        accuracy_training[i] = clf.score(reducedTrainingData.examples, reducedTrainingData.labels)
        accuracy_validation[i] = clf.score(validationData.examples, validationData.labels)
        accuracy_testing[i] = clf.score(testingData.examples, testingData.labels)

    # Plotting accuracies
    def accuracyPlot(accuracyTraining, accuracyTesting, accuracyValidation, title, C):
        # Creating plot with list t in the x-axis and training accuracy in the y-axis
        plt.plot(C, accuracyTraining, 'b', label="Training accuracy")
        # Creating plot with list t in the x-axis and testing accuracy in the y-axis
        plt.plot(C, accuracyTesting, 'r', label="Testing accuracy")
        # Creating plot with list t in the x-axis and testing accuracy in the y-axis
        plt.plot(C, accuracyValidation, 'black', label="Validation accuracy")
        # Logarithmic x axis scale
        plt.xscale('log')
        # Naming the x axis
        plt.xlabel('C parameter value')
        # Naming the y axis
        plt.ylabel('Accuracy')
        # Defining axes limits
        # plt.ylim(0.6, 1)
        # plt.xlim(0, 50)
        # Giving a title to the graph
        plt.title(title)
        # Legend
        plt.legend()
        # Showing the plot
        plt.show()

    # Plotting number of support vectors
    def SVPlot(number_SV, title, c):
        # Creating plot with list t in the x-axis and training accuracy in the y-axis
        plt.plot(c, number_SV, 'b')
        # Logarithmic x axis scale
        plt.xscale('log')
        # Naming the x axis
        plt.xlabel('C parameter value')
        # Naming the y axis
        plt.ylabel('Number of support vectors')
        # Defining axes limits
        # plt.ylim(0.6, 1)
        # plt.xlim(0, 50)
        # Giving a title to the graph
        plt.title(title)
        # Showing the plot
        plt.show()

    accuracyPlot(accuracy_training, accuracy_testing, accuracy_validation, 'Linear Kernel Accuracies', C)

    SVPlot(numberSV, 'Number of support vectors depending on C parameter', C)




