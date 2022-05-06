# ------------------------------------------- WSU - CPT_S 570 - HW3 -----------------------------------------
# ----------------------------------- Code to collect and organize data set -------------------------------------
# ---------------------------------------- Author: Francisco Goncalves ------------------------------------------

import numpy as np

# Open data files
f_trainData = open('data/traindata.txt', 'r')
f_trainLabels = open('data/trainlabels.txt', 'r')
f_testData = open('data/testdata.txt', 'r')
f_testLabels = open('data/testlabels.txt', 'r')
f_stopList = open('data/stoplist.txt', 'r')

# Read data files
trainData = f_trainData.readlines()
trainLabels = f_trainLabels.readlines()
testData = f_testData.readlines()
testLabels = f_testLabels.readlines()
stopList = f_stopList.readlines()

# Close data files
f_trainData.close()
f_trainLabels.close()
f_testData.close()
f_testLabels.close()
f_stopList.close()

# Create vocabulary list of words
vocabulary = []
for i in range(len(trainData)):
    for k in range(len(trainData[i].split())):
        vocabulary.append(trainData[i].split()[k])

# Create list for the stop words
stopWords = []
for i in range(len(stopList)):
    for k in range(len(stopList[i].split())):
        stopWords.append(stopList[i].split()[k])

# Remove duplicated words from vocabulary list
vocabulary = list(dict.fromkeys(vocabulary))

# Remove stop words from vocabulary
vocabulary = list(set(vocabulary) - set(stopWords))

# Organize vocabulary in alphabetical order
vocabulary = sorted(vocabulary)

# Function that defines feature matrix (columns-words, lines-messages). If a message has a certain word the value '1' is
# assigned to that entry
def featureData(data):
    featured_data = np.zeros((len(trainData), len(vocabulary)))
    for i in range(len(data)):
        for k in range(len(vocabulary)):
            for h in range(len(data[i].split())):
                if data[i].split()[h] == vocabulary[k]:
                    featured_data[i, k] = 1
    return featured_data


features = featureData(trainData)


# Defining the lists of labels as a list of integers
for i in range(len(trainLabels)):
    trainLabels[i] = int(trainLabels[i])

for i in range(len(testLabels)):
    testLabels[i] = int(testLabels[i])

# Computing probabilities of each label
prob_y_1 = np.count_nonzero(trainLabels)/len(trainLabels)
prob_y_0 = (len(trainLabels) - np.count_nonzero(trainLabels))/len(trainLabels)

# Function that counts how many times a feature appears in sentences with same label
def n_i_label(label):
    N_i_label = np.zeros(len(vocabulary))
    for d in range(features.shape[1]):
        for t in range(features.shape[0]):
            if features[t, d] == 1 and trainLabels[t] == label:
                N_i_label[d] += 1
    return N_i_label


N_i_1 = n_i_label(1)
N_i_0 = n_i_label(0)

# The fraction of future messages where x_i appears
prob_x_y_1 = (N_i_1 + 1) / (np.count_nonzero(trainLabels) + 2)
# The fraction of wise messages where x_i appears
prob_x_y_0 = (N_i_0 + 1) / ((len(trainLabels)-np.count_nonzero(trainLabels)) + 2)



