# ------------------------------------------- WSU - CPT_S 570 - HW1 -----------------------------------------
# ----------------------- Online Binary Classifier Learning Algorithm - Averaged Perceptron -------------------
# --------------------------------------------- Author: Francisco Goncalves ------------------------------------------
#
# This code is a function that runs the averaged perceptron algorithm for Binary Classification and calculates the
# accuracy and number of mistakes per iteration
#
# INPUTS:
#   T - Number of iterations (integer)
#
# OUTPUTS:
#   accuracy_training_AP - Accuracy on the training data per iteration
#   accuracy_testing_AP - Accuracy on the testing data per iteration
#

from loadingData import D_bin, E_bin
import numpy as np

# Defining the function to be called in the main script
def BinaryAPaccuracy(T):
    print("Please wait a few moments while accuracy for Binary Classification AP is computed...\n")
    # Initializing weights
    w = np.zeros((1, 784))
    w_sum = np.zeros((1, 784))
    # Defining learning rate 'tau'
    tau = 1
    # Prealocating storage output vectors
    accuracy_training_AP = np.zeros((1, T))
    accuracy_testing_AP = np.zeros((1, T))
    # Computing accuracy
    for i in range(T):
        # print("Iteration " + str(i + 1) + " of " + str(T))
        # Defining number of mistakes as zero at the beginning of each iteration
        training_mistakes = 0
        for k in range(D_bin[0].shape[0]):
            # Computing the label prediction 'y_hat = sign(w.x_t)'
            prediction = np.sign(np.inner(w, D_bin[0][k]))
            # If prediction is different from the actual label
            if prediction != D_bin[1][k]:
                # Add one more mistake for this iteration
                training_mistakes = training_mistakes + 1
                # Updating weight vector
                w = w + tau * D_bin[0][k] * D_bin[1][k]
                # Update the weight vector sum
                w_sum = w_sum + w
        # Compute the weight vector average
        w_average = w_sum/training_mistakes
        # Initializing number of correct predicted labels
        correct_testing = 0
        # For all testing examples
        for j in range(E_bin[0].shape[0]):
            # Predict label
            prediction = np.sign(np.inner(w_average, E_bin[0][j]))
            # If prediction is correct
            if prediction == E_bin[1][j]:
                # Add one more correct prediction
                correct_testing = correct_testing + 1
        # Storing accuracies (Total Correct Predictions/Number of testing examples) for testing data for
        # this iteration
        accuracy_testing_AP[0, i] = correct_testing/10000
        # Initializing number of correct predicted labels
        correct_training = 0
        # For all training examples
        for j in range(D_bin[0].shape[0]):
            # Predict label
            prediction = np.sign(np.inner(w_average, D_bin[0][j]))
            # If prediction is correct
            if prediction == D_bin[1][j]:
                # Add one more correct prediction
                correct_training = correct_training + 1
        # Storing accuracies (Total Correct Predictions/Number of training examples) for training data for
        # this iteration
        accuracy_training_AP[0, i] = correct_training/60000
    return accuracy_training_AP, accuracy_testing_AP
