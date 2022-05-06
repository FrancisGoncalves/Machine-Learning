# ------------------------------------------- WSU - CPT_S 570 - HW1 -----------------------------------------
# ------------------------------------------ Code to collect data set -------------------------------------
# ---------------------------------------- Author: Francisco Goncalves ------------------------------------------

# Package to collect data set
from mlxtend.data import loadlocal_mnist
import copy
import numpy as np

# Downloading dataset
trainImages, trainLabels = loadlocal_mnist(images_path='data\\train-images-idx3-ubyte', labels_path='data\\train-labels-idx1-ubyte')
testImages, testLabels = loadlocal_mnist(images_path='data\\t10k-images-idx3-ubyte', labels_path='data\\t10k-labels-idx1-ubyte')

trainLabels = trainLabels.astype('int32')
trainImages = trainImages.astype('int32')
testLabels = testLabels.astype('int32')
testImages = testImages.astype('int32')

# E = testing examples
E = (testImages/255, testLabels)

# D = training examples
D = (trainImages/255, trainLabels)
# print(D)


# Assign the even labels the value 1 and the odd labels the value -1 for the Binary Classification
D_bin = copy.deepcopy(D)
for p in range(D_bin[1].shape[0]):
    if D_bin[1][p] == 0 or D_bin[1][p] == 2 or D_bin[1][p] == 4 or D_bin[1][p] == 6 or D_bin[1][p] == 8:
        D_bin[1][p] = 1
    else:
        D_bin[1][p] = -1

# # Assign the even labels the value 1 and the odd labels the value -1 for the Binary Classification
E_bin = copy.deepcopy(E)
for label in range(E_bin[1].shape[0]):
    if E_bin[1][label] == 0 or E_bin[1][label] == 2 or E_bin[1][label] == 4 or E_bin[1][label] == 6 or E_bin[1][label] == 8:
        E_bin[1][label] = 1
    else:
        E_bin[1][label] = -1

