# ------------------------------------------- WSU - CPT_S 570 - HW2 -----------------------------------------
# ----------------------------------------------- Decision Tree ---------------------------
# ---------------------------------------- Author: Francisco Goncalves ------------------------------------------

from loadingDataDT import training_examples, training_labels, validation_examples, validation_labels, testing_examples, testing_labels
import numpy as np
import copy

def decisionTree():
    print('Please wait a few seconds...')
    # Initializing list to store current examples on each node. This list is organized in the following way:
    # For each layer of tree, this list has the examples present on each node, for example:
    # If the current layer has 2 nodes, then 'examples_nodes_ = [[examples in the 1st node], [examples in the 2nd node]]'
    # First node (root) contains all the examples
    examples_node_ = []
    examples_node_.append(training_examples)
    # Initializing list to store current labels of the examples on each node. The list organized in a similar way to the
    # previous one
    labels_node_ = []
    labels_node_.append(training_labels)
    # Initializing variable that checks if all nodes are pure
    pure = 0
    # Variable to count number of layers of the tree (final layer number will be the depth of the tree)
    layer = 0
    # Initialize list to store selected features with which to split the nodes
    features = []
    # Initialize list to store the threshold in which the chosen feature was split
    splits = []
    tree_nodes = []

    # While all nodes are not pure
    while pure == 0:
        # Current examples per node in this layer
        examples_node = copy.deepcopy(examples_node_)
        # Current labels per node in this layer
        labels_node = copy.deepcopy(labels_node_)
        layer += 1
        # For each node
        for n in range(len(examples_node)):
            # print(labels_node[n])
            # Initialize entropy in order to have something to compare the first one calculated
            lowest_conditional_entropy = 1
            # For each feature
            for f in range(examples_node[n].shape[1]):
                # For each examples
                for e in range(examples_node[n].shape[0]-1):
                    # Initialize variable that counts malignant labels below split
                    m_inf = 0
                    # Initialize variable that counts benignant labels below split
                    b_inf = 0
                    # Initialize variable that counts malignant labels above split
                    m_sup = 0
                    # Initialize variable that counts benignant labels above split
                    b_sup = 0
                    # Initialize variable that counts number of labels below the split
                    inf = 0
                    # Initialize variable that counts number of labels above the split
                    sup = 0
                    # Initialize variables that compute conditional entropy below and above the split
                    conditional_entropy_inf = 0
                    conditional_entropy_sup = 0
                    # Initialize lists to store examples and labels below and above the split
                    inf_examples = []
                    sup_examples = []
                    labels_inf = []
                    labels_sup = []
                    # Go through all examples to compute candidate threshold
                    for k in range(examples_node[n].shape[0]):
                        # Compute threshold
                        split = examples_node[n][e, f] + (examples_node[n][e + 1, f] - examples_node[n][e, f])/2
                        # If feature of example k is below threshold
                        if examples_node[n][k, f] < split:
                            # Add example and its label to the corresponding lists
                            inf_examples.append(examples_node[n][k, :])
                            labels_inf.append(labels_node[n][k])
                            inf += 1
                            # Checking which is the label and increasing the respective variable
                            if labels_node[n][k] == 0:
                                m_inf += 1
                            else:
                                b_inf += 1
                        # If feature of example k is above threshold
                        else:
                            # Add example and its label to the corresponding lists
                            sup_examples.append(examples_node[n][k, :])
                            labels_sup.append(labels_node[n][k])
                            sup += 1
                            # Checking which is the label and increasing the respective variable
                            if labels_node[n][k] == 0:
                                m_sup += 1
                            else:
                                b_sup += 1
                    # Conditions to avoid errors if 'log = 0' or a denominator equals zero
                    if m_inf == 0:
                        ln_m_inf = 0
                    else:
                        ln_m_inf = np.log(m_inf/(m_inf + b_inf))
                    if b_inf == 0:
                        ln_b_inf = 0
                    else:
                        ln_b_inf = np.log(b_inf/(m_inf + b_inf))
                    if (m_inf + b_inf) == 0:
                        conditional_entropy_inf = 0
                    else:
                        conditional_entropy_inf = -((m_inf/(m_inf + b_inf)) * ln_m_inf + (b_inf/(m_inf + b_inf)) * ln_b_inf)
                    if m_sup == 0:
                        ln_m_sup = 0
                    else:
                        ln_m_sup = np.log(m_sup/(m_sup + b_sup))
                    if b_sup == 0:
                        ln_b_sup = 0
                    else:
                        ln_b_sup = np.log(b_sup/(m_sup + b_sup))
                    if (m_sup + b_sup) == 0:
                        conditional_entropy_sup = 0
                    else:
                        conditional_entropy_sup = -((m_sup / (m_sup + b_sup)) * ln_m_sup + (b_sup / (m_sup + b_sup)) * ln_b_sup)
                    # Compute conditional entropy for this feature with this split
                    conditional_entropy = (inf/(inf + sup)) * conditional_entropy_inf + (sup/(inf + sup)) * conditional_entropy_sup
                    # Check if entropy is less than the lowest entropy calculated until this point
                    if conditional_entropy < lowest_conditional_entropy:
                        # Set this conditional entropy as the lowest
                        lowest_conditional_entropy = copy.deepcopy(conditional_entropy)
                        # Save feature and split with lowest entropy
                        selected_feature = f
                        selected_split = split
                        # Assigning values to a different variable for coding purposes
                        sup_examples_ = copy.deepcopy(sup_examples)
                        inf_examples_ = copy.deepcopy(inf_examples)
                        labels_inf_ = copy.deepcopy(labels_inf)
                        labels_sup_ = copy.deepcopy(labels_sup)
            # In the first node of each layer these lists are reinitialized in order to store the data for the current layer
            # but keep the data of the previous so we can keep computing features and splits
            if n == 0:
                examples_node_ = []
                labels_node_ = []
            # Adding selected feature and split to corresponding list
            features.append(selected_feature)
            splits.append(selected_split)
            # Initializing variables to check if all labels of both splitting branches are the same
            count_inf = 1
            count_sup = 1
            # Check if labels of inferior branch are the same
            for i in range(len(labels_inf_)-1):
                if labels_inf_[i] == labels_inf_[i+1]:
                    count_inf += 1
            # Check if labels of superior branch are the same
            for i in range(len(labels_sup_)-1):
                if labels_sup_[i] == labels_sup_[i+1]:
                    count_sup += 1
            # If they are not the same, it means it is an unpure node, so we add the examples to the corresponding list
            if count_inf != len(labels_inf_):
                inf_examples_ = np.array(inf_examples_)
                examples_node_.append(inf_examples_)
                labels_inf_ = np.array(labels_inf_)
                labels_node_.append(labels_inf_)
                tree_nodes.append(labels_inf_)
            # If they are the same, it is a pure node, so we do not add the examples to the list
            # elif count_inf == len(labels_inf_) or len(labels_inf_) == 1:
                # print('PURE NODE INF')
            # Same procedure
            if count_sup != len(labels_sup_):
                sup_examples_ = np.array(sup_examples_)
                examples_node_.append(sup_examples_)
                labels_sup_ = np.array(labels_sup_)
                labels_node_.append(labels_sup_)
                tree_nodes.append(labels_sup_)
            # elif count_sup == len(labels_sup_) or len(labels_sup_) == 1:
                # print('PURE NODE SUP')
        # If the list is empty it means all nodes are pure, so we can end the cycle
        if examples_node_ == []:
            pure = 1


    print('\nThe features selected for splitting the nodes are: ')
    print(features)
    print('\nThe thresholds selected for splitting the nodes are: ')
    print(splits)

    # Function to compute accuracy of the obtained decision tree
    def decisionTreeAccuracy(data, labels_data, data_type):
        predicted_label = np.zeros(data.shape[0])
        for t in range(data.shape[0]):
            if data[t, 22] < 105.2:
                if data[t, 27] < 0.1268:
                    if data[t, 10] < 0.57335:
                        if data[t, 21] < 29.805:
                            predicted_label[t] = 1
                        else:
                            if data[t, 27] < 0.088185:
                                predicted_label[t] = 1
                            else:
                                if data[t, 0] < 12.695:
                                    predicted_label[t] = 0
                                else:
                                    predicted_label[t] = 1
                    else:
                        if data[t, 6] < 0.02755:
                            predicted_label[t] = 0
                        else:
                            predicted_label[t] = 1
                else:
                    if data[t, 24] < 0.16465:
                        if data[t, 23] < 716.7:
                            if data[t, 8] < 0.2613:
                                predicted_label[t] = 1
                            else:
                                predicted_label[t] = 0
                        else:
                            predicted_label[t] = 0
                    else:
                        predicted_label[t] = 0
            else:
                if data[t, 22] < 120.35:
                    if data[t, 1] < 19.69:
                        if data[t, 24] < 0.1344:
                                if data[t, 21] < 28.93:
                                    predicted_label[t] = 1
                                else:
                                    predicted_label[t] = 0
                        else:
                            if data[t, 20] < 16.21:
                                predicted_label[t] = 1
                            else:
                                predicted_label[t] = 0
                    else:
                        predicted_label[t] = 0
                else:
                    predicted_label[t] = 0

        correct = 0
        for t in range(data.shape[0]):
            if predicted_label[t] == labels_data[t]:
                correct += 1

        accuracy_data = correct/data.shape[0]
        print('\nThe accuracy on ' + data_type + ' is ' + str(accuracy_data))

    # Compute validation accuracy
    decisionTreeAccuracy(validation_examples, validation_labels, 'validation data')
    # Compute testing accuracy
    decisionTreeAccuracy(testing_examples, testing_labels, 'testing data')

