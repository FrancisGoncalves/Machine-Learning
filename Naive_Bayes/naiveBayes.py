# ------------------------------------------- WSU - CPT_S 570 - HW3 -----------------------------------------
# ------------------------------------------- Naive Bayes Algorithm ---------------------------------------------
# ----------------------------------------- Author: Francisco Goncalves ------------------------------------------
# -------------------------------------------------- 11/15/2021 ---------------------------------------------------

from loadingData import trainData, trainLabels, testLabels, testData, featureData, prob_x_y_1, \
                        prob_y_0, prob_y_1, prob_x_y_0
print('\nThis is HW 3 of CPT_S 570 Machine Learning.\n')
print('Please wait a few minutes while the Naive Bayes classification is being computed...\n')
# Transforming testing data into features matrix
test_features = featureData(testData)

# Function that calculates product between independent probabilities P(x_i|y)
def calculate_prod_prob(example_features, label):
    prod_p_y_x = 1
    for feature in range(len(example_features)):
        if example_features[feature] == 1 and label == 1:
            prod_p_y_x *= prob_x_y_1[feature]
        elif example_features[feature] == 1 and label == 0:
            prod_p_y_x *= prob_x_y_0[feature]
    return prod_p_y_x

# Function to predict class label
def predict(prod0, prod1):
    prob_y0_x = prob_y_0 * prod0
    prob_y1_x = prob_y_1 * prod1
    if prob_y0_x > prob_y1_x:
        prediction = 0
    else:
        prediction = 1
    return prediction

# Function to compute accuracy of the classifier
def computeAccuracy(data, labels):
    correct = 0
    for message in range(len(data)):
        data_features = featureData(data)
        prod_p_y_x_0 = calculate_prod_prob(data_features[message], 0)
        prod_p_y_x_1 = calculate_prod_prob(data_features[message], 1)
        prediction = predict(prod_p_y_x_0, prod_p_y_x_1)
        if prediction == labels[message]:
            correct += 1
        accuracy = correct/len(data)
    return accuracy


accuracy_training = computeAccuracy(trainData, trainLabels)
accuracy_testing = computeAccuracy(testData, testLabels)

file = open("output.txt", "w")
file.write('The training accuracy is ' + str(accuracy_training) + '.\n')
file.write('The testing accuracy is ' + str(accuracy_testing) + '.\n')
file.close()

print('CLASSIFICATION FINISHED.\n')
print("You can check the results of the Naive Bayes classifier in 'output.txt' file.\n")
