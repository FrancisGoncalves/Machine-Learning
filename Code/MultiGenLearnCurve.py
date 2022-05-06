# ------------------------------------------- WSU - CPT_S 570 - HW1 -----------------------------------------
# --------------------- General Learning Curve for Multi Class Classification - Standard Perceptron -------------------
# ----------------------------------------- Author: Francisco Goncalves ------------------------------------------
#
# This code is a function that runs the perceptron algorithm for Multi-Class Classification and calculates the General
# Learning Curve (accuracy per number of example used during training)
#
# INPUTS:
#   T - Number of iterations (integer)
#
# OUTPUTS:
#   accuracy_multiSP_testing - Accuracy on the testing data per iteration

#
from loadingData import D, E
import numpy as np

# Defining the function to be called in the main script
def MultiGenLearnCurve(T):

    print("\n This can take 2 or 3 hours to compute. You can go get some coffee and come back later...\n")
    # Defining function that allocates example 'xt' to a particular position in a vector according to label 'yt'
    def F(xt, yt):
        d = np.zeros((1, 10 * int(xt.shape[0])))
        d[0, yt * int(xt.shape[0]):784 * (yt + 1)] = xt
        return d
    # Defining learning rate 'tau'
    tau = 1
    # Prealocating scores vector
    scores = np.zeros((1, 10))
    # Initializing examples
    examples = 0
    # Prealocating storage output vector
    accuracy_multiSP_testing = np.zeros((1, 600))
    # Learning algorithm
    while examples < D[0].shape[0]:
        # Increasing the number of training examples dataset by 100
        examples = examples + 100
        # print("Examples = " + str(examples))
        # Initializing weights
        w = np.zeros((1, 7840))
        for i in range(T):
            # Defining number of mistakes as zero at the beginning of each iteration
            training_mistakes = 0
            # print("Iteration " + str(i + 1) + " of " + str(T))
            for k in range(examples):
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
        accuracy_multiSP_testing[0, int(examples/100 - 1)] = correct_testing / 10000
    return accuracy_multiSP_testing