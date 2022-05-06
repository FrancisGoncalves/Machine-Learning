# ------------------------------------------- WSU - CPT_S 570 - HW1 -----------------------------------------
# --------------------- Online Multi-Class Classifier Learning Algorithm - Standard Perceptron -------------------
# ----------------------------------------- Author: Francisco Goncalves ------------------------------------------
#
# This code is a function that runs the perceptron algorithm for Multi-Class Classification and calculates the accuracy
# and number of mistakes per iteration
#
# INPUTS:
#   T - Number of iterations (integer)
#   accuracy_trigger - This variable activates or deactivates the computation of accuracy in order to save computational
#                      time in case we just want the number of mistakes (0/1)
# OUTPUTS:
#   accuracy_multiSP_training - Accuracy on the training data per iteration
#   accuracy_multiSP_testing - Accuracy on the testing data per iteration
#   mistakes_multi_SP - Mistakes per training iteration
#
from loadingData import D, E
import numpy as np

# Defining the function to be called in the main script
def MultiSP(T, accuracy_trigger):
    if accuracy_trigger == 1:
        print("Please wait a few moments while accuracy for Multi-Class Classification SP is computed...\n")
    else:
        print("Please wait a few moments while Online Learning curve for Multi-Class Classification SP is computed...\n")
    # Defining function that allocates example 'xt' to a particular position in a vector according to label 'yt'
    def F(xt, yt):
        d = np.zeros((1, 10 * int(xt.shape[0])))
        d[0, yt * int(xt.shape[0]):784 * (yt + 1)] = xt
        return d
    # Initializing weights
    w = np.zeros((1, 7840))
    # Defining learning rate 'tau'
    tau = 1
    # Prealocating scores vector
    scores = np.zeros((1, 10))
    # Prealocating storage output vectors
    accuracy_multiSP_training = np.zeros((1, T))
    accuracy_multiSP_testing = np.zeros((1, T))
    mistakes_multi_SP = np.zeros((1, T))
    # Learning algorithm
    for i in range(T):
        # Defining number of mistakes as zero at the beginning of each iteration
        training_mistakes = 0
        # print("Iteration " + str(i + 1) + " of " + str(T))
        for k in range(D[0].shape[0]):
            for y in range(10):
                # Computing scores for each label (0, 1, ..., 9)
                scores[:, y] = np.inner(w[0, y * int(D[0].shape[1]):784 * (y + 1)], D[0][k, :])
            # Computing the label prediction (label with highest score)
            prediction = np.argmax(scores)
            # If prediction is different from the actual label
            if prediction != D[1][k]:
                # Add one more mistake for this iteration
                training_mistakes = training_mistakes + 1
                # Updating weight vector
                w = w + tau * (F(D[0][k], D[1][k]) - F(D[0][k], prediction))
        # Storing number of mistakes for this iteration
        mistakes_multi_SP[0, i] = training_mistakes
        # If the accuracy trigger is activated
        if accuracy_trigger == 1:
            # Initializing number of correct predicted labels
            correct_training = 0
            # For all training examples
            for j in range(D[0].shape[0]):
                for y in range(10):
                    scores[:, y] = np.inner(w[0, y * int(D[0].shape[1]):784 * (y + 1)], D[0][j, :])
                # Predict label
                prediction = np.argmax(scores)
                # If prediction is correct
                if prediction == D[1][j]:
                    # Add one more correct prediction
                    correct_training = correct_training + 1
            # Storing accuracies (Total Correct Predictions/Number of training examples) for training data for
            # this iteration
            accuracy_multiSP_training[0, i] = correct_training/60000
            # Initializing number of correct predicted labels
            correct_testing = 0
            # For all testing examples
            for j in range(E[0].shape[0]):
                for y in range(10):
                    scores[:, y] = np.inner(w[0, y * int(E[0].shape[1]):784 * (y + 1)], E[0][j, :])
                # Predict label
                prediction = np.argmax(scores)
                # If prediction is correct
                if prediction == E[1][j]:
                    # Add one more correct prediction
                    correct_testing = correct_testing + 1
            # Storing accuracies (Total Correct Predictions/Number of testing examples) for testing data for
            # this iteration
            accuracy_multiSP_testing[0, i] = correct_testing/10000
    return accuracy_multiSP_training, accuracy_multiSP_testing, mistakes_multi_SP
