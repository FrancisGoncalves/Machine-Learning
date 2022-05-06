# ------------------------------------------- WSU - CPT_S 570 - HW1 -----------------------------------------
# --------------------- Online Binary Classifier Learning Algorithm - Passive-Agressive -------------------
# --------------------------------------------- Author: Francisco Goncalves ------------------------------------------
#
# This code is a function that runs the Passive Agressive algorithm for Binary Classification and calculates the
# accuracy and number of mistakes per iteration
#
# INPUTS:
#   T - Number of iterations (integer)
#   accuracy_trigger - This variable activates or deactivates the computation of accuracy in order to save computational
#                      time in case we just want the number of mistakes (0/1)
# OUTPUTS:
#   accuracy_training_PA - Accuracy on the training data per iteration
#   accuracy_testing_PA - Accuracy on the testing data per iteration
#   mistakes_bin_PA - Mistakes per training iteration
#
from loadingData import D_bin, E_bin
import numpy as np

# Defining the function to be called in the main script
def BinaryPA(T, accuracy_trigger):
    if accuracy_trigger == 1:
        print("Please wait a few moments while accuracy for Binary Classification PA is computed...\n")
    else:
        print("Please wait a few moments while Online Learning curve for Binary Classification PA is computed...\n")
    # Initializing weights
    w = np.zeros((1, 784))
    # Initializing storage output vectors
    accuracy_training_PA = np.zeros((1, T))
    accuracy_testing_PA = np.zeros((1, T))
    mistakes_bin_PA = np.zeros((1, T))
    # Learning algorithm
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
                # Update learning rate 'tau'
                tau = (1-D_bin[1][k]*(np.inner(w, D_bin[0][k])))/((np.linalg.norm(D_bin[0][k]))**2)
                # Update weight vector
                w = w + tau * D_bin[0][k] * D_bin[1][k]
        # Storing number of mistakes for this iteration
        mistakes_bin_PA[0, i] = training_mistakes
        # If the accuracy trigger is activated
        if accuracy_trigger == 1:
            # Initializing number of correct predicted labels
            correct_testing = 0
            # For all testing examples
            for j in range(E_bin[0].shape[0]):
                # Predict label
                prediction = np.sign(np.inner(w, E_bin[0][j]))
                # If prediction is correct
                if prediction == E_bin[1][j]:
                    # Add one more correct prediction
                    correct_testing = correct_testing + 1
            # Storing accuracies (Total Correct Predictions/Number of testing examples) for testing data for
            # this iteration
            accuracy_testing_PA[0, i] = correct_testing/10000
            # Initializing number of correct predicted labels
            correct_training = 0
            # For all training examples
            for j in range(D_bin[0].shape[0]):
                # Predict label
                prediction = np.sign(np.inner(w, D_bin[0][j]))
                # If prediction is correct
                if prediction == D_bin[1][j]:
                    # Add one more correct prediction
                    correct_training = correct_training + 1
            # Storing accuracies (Total Correct Predictions/Number of training examples) for training data for
            # this iteration
            accuracy_training_PA[0, i] = correct_training/60000
    return accuracy_training_PA, accuracy_testing_PA, mistakes_bin_PA

