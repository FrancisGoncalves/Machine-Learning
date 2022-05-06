# ------------------------------------------- WSU - CPT_S 570 - HW2 -----------------------------------------
# ----------------------------------- Code to collect and organize data set -------------------------------------
# ---------------------------------------- Author: Francisco Goncalves ------------------------------------------

# Package to collect data set
from mlxtend.data import loadlocal_mnist

# Downloading dataset
trainImages, trainLabels = loadlocal_mnist(images_path='data/train-images-idx3-ubyte', labels_path='data/train-labels-idx1-ubyte')
testImages, testLabels = loadlocal_mnist(images_path='data/t10k-images-idx3-ubyte', labels_path='data/t10k-labels-idx1-ubyte')

# Converting all 'uint8' type data to integer
trainLabels = trainLabels.astype('int32')
trainImages = trainImages.astype('int32')
testLabels = testLabels.astype('int32')
testImages = testImages.astype('int32')

# Class of whole Training Data
class TrainingData:
    def __init__(self):
        self.examples = trainImages/255
        self.labels = trainLabels


trainingData = TrainingData()

# Class for reduced training data (training data - validation data)
class ReducedTrainingData:
    def __init__(self):
        self.examples = trainImages[0:int(0.8 * trainImages.shape[0]), :]/255
        self.labels = trainLabels[0:int(0.8 * trainLabels.shape[0])]


reducedTrainingData = ReducedTrainingData()

# Reduced training data for Kernelized Perceptron
class ReducedTrainingDataKP:
    def __init__(self):
        self.examples = trainImages[0:int(0.3 * trainImages.shape[0]), :] / 255
        self.labels = trainLabels[0:int(0.3 * trainLabels.shape[0])]


reducedTrainingDataKP = ReducedTrainingDataKP()

# Class for validation data
class ValidationData:
    def __init__(self):
        self.examples = trainImages[int(0.8 * trainImages.shape[0]):, :]/255
        self.labels = trainLabels[int(0.8 * trainImages.shape[0]):]


validationData = ValidationData()

# Class for validation data for Kernelized Perceptron
class ValidationDataKP:
    def __init__(self):
        self.examples = trainImages[int(0.9 * trainImages.shape[0]):, :]/255
        self.labels = trainLabels[int(0.9 * trainImages.shape[0]):]


validationDataKP = ValidationDataKP()

# Class for testing data
class TestingData:
    def __init__(self):
        self.examples = testImages[0:int(testImages.shape[0]), :]/255
        self.labels = testLabels[0:int(testImages.shape[0])]


testingData = TestingData()
# Class for testing data for Kernelized Perceptron
class TestingDataKP:
    def __init__(self):
        self.examples = testImages[0:int(0.5 * testImages.shape[0]), :]/255
        self.labels = testLabels[0:int(0.5 * testImages.shape[0])]


testingDataKP = TestingDataKP()