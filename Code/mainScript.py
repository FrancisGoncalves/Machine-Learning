# ------------------------------------------- WSU - CPT_S 570 - HW1 -----------------------------------------
# ------------------------------------------------- Main Script ---------------------------------------------
# ----------------------------------------- Author: Francisco Goncalves ------------------------------------------
# -------------------------------------------------- 09/23/2021 ---------------------------------------------------

import matplotlib.pyplot as plt
from BinaryPA import BinaryPA
from BinarySP import BinarySP
from BinaryAPaccuracy import BinaryAPaccuracy
from MultiSP import MultiSP
from MultiPA import MultiPA
from MultiAPaccuracy import MultiAPaccuracy
from BinGenLearnCurve import BinGenLearnCurve
from MultiGenLearnCurve import MultiGenLearnCurve
import numpy as np

# Welcoming message
print("\nWelcome to Homework 1 of CPT_S 570 - Machine Learning!")

# Function that plots Online Learning Curves
def OLPlot(mistakes, title, color, T):
    # Prealocating list t
    t = np.zeros((1, T))
    # Creating list with numbers from 1 to T (1,2, ..., T)
    t[0, :] = list(range(1, T + 1))
    # Creating plot with list t in the x-axis and number of mistakes in the y-axis
    plt.plot(t[0], mistakes[0], color)
    # Naming the x axis
    plt.xlabel('Number of training iterations')
    # Naming the y axis
    plt.ylabel('Mistakes')
    # Giving a title to the graph
    plt.title(title)
    # Showing the plot
    plt.show()

# Function that plots accuracy curves
def accuracyPlot(accuracy_train, accuracy_tests, title, T):
    # Prealocating list t
    t = np.zeros((1, T))
    # Creating list with numbers from 1 to T (1,2, ..., T)
    t[0, :] = list(range(1, T + 1))
    # Creating plot with list t in the x-axis and training accuracy in the y-axis
    plt.plot(t[0], accuracy_train[0], 'b', label="Training accuracy")
    # Creating plot with list t in the x-axis and testing accuracy in the y-axis
    plt.plot(t[0], accuracy_tests[0], 'r', label="Testing accuracy")
    # Naming the x axis
    plt.xlabel('Number of training iterations')
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

# Function that plots the General Learning Curve
def genLearnCurve(accuracy, title):
    # Prealocating variable examples
    examples = np.zeros((1, 600))
    # Creating list 'examples' from 1 to 60000 with increments of 100, i.e. examples = (100, 200, ..., 60000)
    examples[0, :] = list(range(1, 60000 + 1, 100))
    # Creating the plot with the number of examples in the x-axis and the Accuracy in the y-axis
    plt.plot(examples[0], accuracy[0], 'b')
    # naming the x axis
    plt.xlabel('Number of examples')
    # naming the y axis
    plt.ylabel('Accuracy')
    # Defining axes limits
    #plt.ylim(0.6, 1)
    # Giving a title to the graph
    plt.title(title)
    # function to show the plot
    plt.show()

# Function that shows the menu for Binary Classification options
def binClassMenu():
    ans_bin = True
    # User gets to choose an option from the menu which will give him one of the HW1 curves
    while ans_bin:
        print("""
        1.Compute online learning curve for Standard Perceptron
        2.Compute online learning curve for Passive-Agressive
        3.Compute accuracy curves for Standard Perceptron
        4.Compute accuracy curves for Passive Agressive
        5.Compute accuracy curves for Averaged Perceptron
        6.Compute General Learning Curve for Standard Perceptron
        7.Back
        """)
        ans_bin = input("What would you like to compute? ")
        # Option 1 shows the Online Learning Curve for Binary Classification using SP
        if ans_bin == "1":
            # If the program already computed the mistakes, if the user chooses this option again, it will not compute
            # them again and will only show the graph
            if 'mistakes_bin_SP' not in locals():
                # Calculating mistakes
                _, _, mistakes_bin_SP = BinarySP(50, 0)
            # Computing Online Learning Curve
            OLPlot(mistakes_bin_SP, 'Online learning curve for Binary Classification using Standard Perceptron', 'b', 50)
        # Option 2 shows the Online Learning Curve for Binary Classification using PA
        elif ans_bin == "2":
            # If the program already computed the mistakes, if the user chooses this option again, it will not compute
            # them again and will only show the graph
            if 'mistakes_bin_PA' not in locals():
                _, _, mistakes_bin_PA = BinaryPA(50, 0)
            # Computing Online Learning Curve
            OLPlot(mistakes_bin_PA, 'Online learning curve for Binary Classification using Passive Agressive', 'r', 50)
        # Option 3 shows the Accuracy Curve for Binary Classification using SP
        elif ans_bin == "3":
            # If the program already computed the accuracies, even if the user chooses this option again, it will not
            # compute them again and will only show the graph
            if 'accuracy_training_SP' not in locals():
                accuracy_training_SP, accuracy_testing_SP, _ = BinarySP(20, 1)
            # Computing Accuracy Curve
            accuracyPlot(accuracy_training_SP, accuracy_testing_SP, "Accuracy of Binary Classidication - Standard Perceptron Algorithm", 20)
        # Option 4 shows the Accuracy Curve for Binary Classification using PA
        elif ans_bin == "4":
            # If the program already computed the accuracies, even if the user chooses this option again, it will not
            # compute them again and will only show the graph
            if 'accuracy_training_PA' not in locals():
                accuracy_training_PA, accuracy_testing_PA, _ = BinaryPA(20, 1)
            # Computing Accuracy Curve
            accuracyPlot(accuracy_training_PA, accuracy_testing_PA, "Accuracy of Binary Classification - Passive Agressive Algorithm", 20)
        # Option 5 shows the Accuracy Curve for Binary Classification using AP
        elif ans_bin == "5":
            # If the program already computed the accuracies, even if the user chooses this option again, it will not
            # compute them again and will only show the graph
            if 'accuracy_training_AP' not in locals():
                accuracy_training_AP, accuracy_testing_AP = BinaryAPaccuracy(20)
            # Computing Accuracy Curve
            accuracyPlot(accuracy_training_AP, accuracy_testing_AP, "Accuracy of Binary Classification - Averaged Perceptron Algorithm", 20)
        # Option 6 shows the General Learning Curve for Binary Classification using SP
        elif ans_bin == "6":
            # If the program already computed the accuracies, even if the user chooses this option again, it will not
            # compute them again and will only show the graph
            if 'accuracy_GLC_bin' not in locals():
                accuracy_GLC_bin = BinGenLearnCurve(20)
            # Computing General Learning Curve
            genLearnCurve(accuracy_GLC_bin, "General Learning Curve - Binary Classification SP")
        # Option to go back to previous menu
        elif ans_bin == "7":
            break
        # If the user selects something that is not on the menu, an error message appears
        elif ans_bin != "":
            print("\n INVALID OPTION! TRY AGAIN")

# Function that shows the menu for Multi-Class Classification options
def multiClassMenu():
    ans_multi = True
    # User gets to choose an option from the menu which will give him one of the HW1 curves
    while ans_multi:
        print("""
        1.Compute online learning curve for Standard Perceptron
        2.Compute online learning curve for Passive-Agressive
        3.Compute accuracy curves for Standard Perceptron
        4.Compute accuracy curves for Passive Agressive
        5.Compute accuracy curves for Averaged Perceptron
        6.Compute General Learning curve
        7.Back
        """)
        ans_multi = input("What would you like to compute? ")
        # Option 1 shows the Online Learning Curve for Multi-Class Classification using SP
        if ans_multi == "1":
            # 'Mentioned previously'
            if 'mistakes_multi_SP' not in locals():
                _, _, mistakes_multi_SP = MultiSP(50, 0)
            # Computes the Online Learning Curve
            OLPlot(mistakes_multi_SP, 'Online learning curve for Multi-Class Classification using Standard Perceptron', 'b', 50)
         # Option 2 shows the Online Learning Curve for Multi-Class Classification using PA
        elif ans_multi == "2":
            # 'Mentioned previously'
            if 'mistakes_multi_PA' not in locals():
                _, _, mistakes_multi_PA = MultiPA(50, 0)
            # Computes the Online Learning Curve
            OLPlot(mistakes_multi_PA, 'Online learning curve for Multi-Class Classification using Passive Agressive', 'r', 50)
        # Option 3 shows the Accuracy Curve for Multi-Class Classification using SP
        elif ans_multi == "3":
            # 'Mentioned previously'
            if 'accuracy_multiSP_training' not in locals():
                accuracy_multiSP_training, accuracy_multiSP_testing, _ = MultiSP(20, 1)
            # Computes the Accuracy Curve
            accuracyPlot(accuracy_multiSP_training, accuracy_multiSP_testing, "Accuracy of Multi-Class Classification - Standard Perceptron Algorithm", 20)
        # Option 4 shows the Accuracy Curve for Multi-Class Classification using PA
        elif ans_multi == "4":
            # 'Mentioned previously'
            if 'accuracy_multiPA_training' not in locals():
                accuracy_multiPA_training, accuracy_multiPA_testing, _ = MultiPA(20, 1)
            # Computes the Accuracy Curve
            accuracyPlot(accuracy_multiPA_training, accuracy_multiPA_testing, "Accuracy of Multi-Class Classification - Passive Agressive Algorithm", 20)
        # Option 5 shows the Accuracy Curve for Multi-Class Classification using AP
        elif ans_multi == "5":
            # 'Mentioned previously'
            if 'accuracy_multiAP_training' not in locals():
                accuracy_multiAP_training, accuracy_multiAP_testing = MultiAPaccuracy(20)
            # Computes the Accuracy Curve
            accuracyPlot(accuracy_multiAP_training, accuracy_multiAP_testing, "Accuracy of Multi-Class Classification - Averaged Perceptron Algorithm", 20)
        # Option 6 shows the General Learning Curve for Multi-Class Classification using SP
        elif ans_multi == "6":
            # 'Mentioned previously'
            if 'accuracy_GLC_multi' not in locals():
                accuracy_GLC_multi = MultiGenLearnCurve(20)
            # Computes the General Learning Curve
            genLearnCurve(accuracy_GLC_multi, "General Learning Curve - Multi-Class Classification SP")
        # Option to go back to previous menu
        elif ans_multi == "7":
            break
        # If the user selects something that is not on the menu, an error message appears
        elif ans_multi != "":
            print("\n INVALID OPTION! TRY AGAIN")

# Main menu
ans = True
# The user gets to choose between Binary or Multi-Class Classification
while ans:
    print("""
    1.Binary Classification
    2.Multi-Class Classification
    3.Quit
    """)
    ans = input("What type of classification would you like to look into? ")
    # Option 1 returns the menu for Binary Classification
    if ans == "1":
        binClassMenu()
    # Option 2 returns the menu for Multi-Class Classification
    elif ans == "2":
        multiClassMenu()
    # Option to quit the program
    elif ans == "3":
        break
    # If the user selects something that is not on the menu, an error message appears
    elif ans != "":
        print("\n INVALID OPTION! TRY AGAIN")
