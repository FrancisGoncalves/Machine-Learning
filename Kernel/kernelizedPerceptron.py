# ------------------------------------------- WSU - CPT_S 570 - HW2 -------------------------------------------------
# ------------------------------------------- Kernelized Perceptron -------------------------------------------------
# ---------------------------------------- Author: Francisco Goncalves ----------------------------------------------

import numpy as np
from loadingData import reducedTrainingDataKP, validationDataKP, testingDataKP
import matplotlib.pyplot as plt

def kernelizedPerceptron():
    print('\nThis may take several hours. You can come back later... I will print the number of the current iteration'
          ' so you can keep track.')
    # Initialize dual weights (alphas) for all classes and examples
    alpha = np.zeros(reducedTrainingDataKP.examples.shape[0] * 10)
    # Initialize scores and mistakes during iterations
    scores = np.zeros(10)
    mistakes_training = np.zeros(5)
    # Initialize vectors to store Kernel values that remain the same in all iterations
    y_Kernel_training = np.zeros(reducedTrainingDataKP.examples.shape[0] * reducedTrainingDataKP.examples.shape[0])
    y_Kernel_validation = np.zeros(reducedTrainingDataKP.examples.shape[0] * validationDataKP.examples.shape[0])
    y_Kernel_testing = np.zeros(reducedTrainingDataKP.examples.shape[0] * testingDataKP.examples.shape[0])
    # For each iteration
    for i in range(5):
        print(i)
        # Initialize sum at the beginning of each iteration
        summ = 0
        for g in range(reducedTrainingDataKP.examples.shape[0]):
            # For each class
            for y in range(10):
                if y > 0:
                    summ = 0 # I think I should probably be fine just with this sum initialization but without the other one
                            # I was having some issues
                # Compute the score for each class
                for k in range(reducedTrainingDataKP.examples.shape[0]):
                    # The Kernels are only computed during the first iteration and only for the first class since they don't change
                    if i == 0 and y == 0:
                        # Calculating quadratic polynomial Kernel
                        y_Kernel_training[g * reducedTrainingDataKP.examples.shape[0] + k] = reducedTrainingDataKP.labels[k] * (1 + round(float(np.inner(reducedTrainingDataKP.examples[k], reducedTrainingDataKP.examples[g])), 2)) ** 2
                    summ += alpha[y * reducedTrainingDataKP.examples.shape[0] + k] * y_Kernel_training[g * reducedTrainingDataKP.examples.shape[0] + k]
                # Storing the score for this class
                scores[y] = summ
            # Compute prediction, which is the class with the highest score
            prediction = np.argmax(scores)
            # If prediction is wrong
            if prediction != reducedTrainingDataKP.labels[g]:
                # Count one more mistake
                mistakes_training[i] = mistakes_training[i] + 1
                # Update alpha values (increase correct class weight and decrease wrong predicted class weight)
                alpha[reducedTrainingDataKP.labels[g] * reducedTrainingDataKP.examples.shape[0] + g] += 1
                alpha[prediction * reducedTrainingDataKP.examples.shape[0] + g] -= 1

    # Computing accuracy on training data
    correct = 0
    for j in range(reducedTrainingDataKP.examples.shape[0]):
        summ = 0
        for y in range(10):
            if y > 0:
                summ = 0
            for k in range(reducedTrainingDataKP.examples.shape[0]):
                # The Kernels are obtained from the previous training
                summ += alpha[y * reducedTrainingDataKP.examples.shape[0] + k] * y_Kernel_training[j * reducedTrainingDataKP.examples.shape[0] + k]
            scores[y] = summ
        # Compute prediction
        prediction = np.argmax(scores)
        # If prediction is correct
        if prediction == reducedTrainingDataKP.labels[j]:
            # Increase number of correct predictions
            correct += 1
    # Calculating accuracy
    accuracy_training = correct/reducedTrainingDataKP.examples.shape[0]
    print('\nThe training accuracy is: ' + str(accuracy_training))

    # Computing accuracy on validation data
    correct = 0
    for j in range(validationDataKP.examples.shape[0]):
        summ = 0
        for y in range(10):
            if y > 0:
                summ = 0
            for k in range(reducedTrainingDataKP.examples.shape[0]):
                # Again the Kernels are only computed for the first class, since for the other ones it is the same
                if y == 0:
                    y_Kernel_validation[j * reducedTrainingDataKP.examples.shape[0] + k] = reducedTrainingDataKP.labels[k] * (1 + round(float(np.inner(reducedTrainingDataKP.examples[k], validationDataKP.examples[j])), 2)) ** 2
                summ += alpha[y * reducedTrainingDataKP.examples.shape[0] + k] * y_Kernel_validation[j * reducedTrainingDataKP.examples.shape[0] + k]
            scores[y] = summ
        # Compute prediction
        prediction = np.argmax(scores)
        # If prediction is correct
        if prediction == validationDataKP.labels[j]:
            # Increase number of correct predictions
            correct += 1
    # Calculating accuracy
    accuracy_validation = correct/validationDataKP.examples.shape[0]
    print('\nThe validation accuracy is: ' + str(accuracy_validation))

    # Computing accuracy on testing data
    correct = 0
    for j in range(testingDataKP.examples.shape[0]):
        summ = 0
        # For all classes
        for y in range(10):
            if y > 0:
                summ = 0
            # For all testing examples
            for k in range(reducedTrainingDataKP.examples.shape[0]):
                if y == 0:
                    # Computing Kernels
                    y_Kernel_testing[j * reducedTrainingDataKP.examples.shape[0] + k] = reducedTrainingDataKP.labels[k] * (
                                1 + round(float(np.inner(reducedTrainingDataKP.examples[k], testingDataKP.examples[j])),
                                          2)) ** 2
                summ += alpha[y * reducedTrainingDataKP.examples.shape[0] + k] * y_Kernel_testing[j * reducedTrainingDataKP.examples.shape[0] + k]
            scores[y] = summ
        # Compute prediction
        prediction = np.argmax(scores)
        # If prediction is correct
        if prediction == testingDataKP.labels[j]:
            # Increase number of correct predictions
            correct += 1
    # Compute accuracy
    accuracy_testing = correct/testingDataKP.examples.shape[0]
    print('\nThe testing accuracy is: ' + str(accuracy_testing))


    def OLPlot(mistakes, title, color, T):
        # Prealocating list t
        t = np.zeros((1, T))
        # Creating list with numbers from 1 to T (1,2, ..., T)
        t[0, :] = list(range(1, T + 1))
        # Creating plot with list t in the x-axis and number of mistakes in the y-axis
        plt.plot(t[0], mistakes, color)
        # Naming the x axis
        plt.xlabel('Number of training iterations')
        # Naming the y axis
        plt.ylabel('Mistakes')
        # Giving a title to the graph
        plt.title(title)
        # Showing the plot
        plt.show()

    # Plot the graph
    OLPlot(mistakes_training, 'Number of mistakes per iteration', 'b', 5)



