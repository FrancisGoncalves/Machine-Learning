# ------------------------------------------- WSU - CPT_S 570 - HW1 -----------------------------------------
# --------------------- General Learning Curve for Binary Classifier - Standard Perceptron -------------------
# ----------------------------------------- Author: Francisco Goncalves ------------------------------------------
#
# This code is a function that runs the perceptron algorithm for Binary Classification and calculates the General
# Learning Curve (accuracy per number of example used during training)
#
# INPUTS:
#   T - Number of iterations (integer)
#
# OUTPUTS:
#   accuracy_testing_SP - Accuracy on the testing data per number of examples in the dataset


from loadingData import D_bin, E_bin
import numpy as np

# Defining the function to be called in the main script
def BinGenLearnCurve(T):
    print("\n This will take a while. You can go get some coffee and come back later...\n")
    # Defining learning rate 'tau'
    tau = 1
    # Prealocating storage output vectors
    accuracy_testing_SP = np.zeros((1, 600))
    # Prealocating variable 'examples'
    examples = 0
    # Computing accuracy
    while examples < D_bin[0].shape[0]:
        # Increasing the number of training examples dataset by 100
        # examples = examples + 100
        print("Examples: " + str(examples))
        # Initializing weights
        w = np.zeros((1, 784))
        # Doing 20 iterations per each set of training examples
        for i in range(T):
            # print("Iteration " + str(i + 1) + " of " + str(T))
            # Defining number of mistakes as zero at the beginning of each iteration
            training_mistakes = 0
            for k in range(examples):
                # Computing the label prediction 'y_hat = sign(w.x_t)'
                prediction = np.sign(np.inner(w, D_bin[0][k]))
                if prediction != D_bin[1][k]:
                    # Add one more mistake for this iteration
                    training_mistakes = training_mistakes + 1
                    # Updating weight vector
                    w = w + tau * D_bin[0][k] * D_bin[1][k]
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
        accuracy_testing_SP[0, int(examples/100 - 1)] = correct_testing/10000
    return accuracy_testing_SP
