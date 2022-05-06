# ------------------------------------------- WSU - CPT_S 570 - HW2 -----------------------------------------
# ---------------------- Code to collect and organize data set for Decision Tree algorithm ---------------------------
# ---------------------------------------- Author: Francisco Goncalves ------------------------------------------

import csv
import numpy as np

# Collect data from csv file and store it as a list of examples and labels
with open('data/wdbc.data', 'r') as train_file:
    trainReader = csv.reader(train_file)
    identification = []
    labels = []
    examples = []
    for row in trainReader:
        identification.append(row[0])
        labels.append(row[1])
        examples.append(row[2:])

# Transforming lists into arrays
examples = np.array(examples)
labels = np.array(labels)
identification = np.array(identification)

identification = identification.astype('int32')
examples = examples.astype('float')

# Converting labels 'M' and 'B' into 0 and 1 respectively, so they are easier to work with
for n in range(labels.shape[0]):
    if labels[n] == 'B':
        labels[n] = '1'
    else:
        labels[n] = '0'
labels = labels.astype('int32')

# Defining training, validation and testing datasets
training_examples = examples[0:int(0.7 * examples.shape[0]), :]
training_labels = labels[0:int(0.7 * labels.shape[0])]
validation_examples = examples[int(0.7 * labels.shape[0]):int(0.9 * examples.shape[0]), :]
validation_labels = labels[int(0.7 * labels.shape[0]):int(0.9 * examples.shape[0])]
testing_examples = examples[int(0.9 * examples.shape[0]):, :]
testing_labels = labels[int(0.9 * examples.shape[0]):]