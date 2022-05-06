# ------------------------------------------- WSU - CPT_S 570 - HW2 -----------------------------------------
# ------------------------------------------------- Main Script ---------------------------------------------
# ----------------------------------------- Author: Francisco Goncalves ------------------------------------------
# -------------------------------------------------- 10/22/2021 ---------------------------------------------------


from linearKernel import linearKernel
from linearKernelBestC import linearKernelBestC
from polynomialKernel import polynomialKernel
from kernelizedPerceptron import kernelizedPerceptron
from decisionTree import decisionTree

# Welcoming message
print("\nWelcome to Homework 2 of CPT_S Machine Learning!")

# Main menu
ans = True
# The user gets to choose between Binary or Multi-Class Classification
while ans:
    print("""
    1.Linear Kernel (find best C parameter)
    2.Linear Kernel using best C parameter and entire training dataset
    3.Polynomial Kernels
    4.Kernelized Perceptron
    5.Decision Tree
    6.Quit
    """)
    ans = input("What type of classification would you like to look into? ")
    # Option 1 computes linear Kernel to get best parameter C
    if ans == "1":
        linearKernel()
    # Option 2 computes accuracy using whole training data and best parameter C
    elif ans == "2":
        linearKernelBestC()
    # Option 3 computes accuracies for polynomial Kernel of degree 2,3 and 4
    elif ans == "3":
        polynomialKernel()
    # Option 4 computes accuracies and learning curve for Kernelized perceptron
    elif ans == "4":
        kernelizedPerceptron()
    # Option 5 computes accuracy of decision tree algorithm on validation and testing data
    elif ans == "5":
        decisionTree()
    # Option to quit the program
    elif ans == "6":
        break
    # If the user selects something that is not on the menu, an error message appears
    elif ans != "":
        print("\n INVALID OPTION! TRY AGAIN")
