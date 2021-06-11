# kNN algorithm
# @author Peter Robe

import pandas as pd
import numpy as np
from scipy import spatial

class Neighbor:
  def __init__(self, label, dist):
    self.label = label
    self.dist = dist

##
# A Tally is a unique label and the number of instances there are among the nearest neighbors
#
class Tally:
    def __init__(self, label, count):
        self.label = label
        self.count = count

##
# @param data The DataFrame of data
# @param label_col The index of the column that contains the labels
# @param n_folds The number folds
# @param k_neighbors The number of neighbors to use when predicting the label
# @return float The average accuracy of all the folds
#
def test(data, label_col, n_folds, k_neighbors):

    # Store all the accuracy values for each fold to average later
    accuracies = list()

    # Subdivide the data
    for fold_num in range(n_folds):
        # The first index in the data DataFrame that is considered testing data
        test_start = int(len(data.index) * (fold_num / n_folds))
        # The last index in the data DataFrame that is considered testing data PLUS 1
        test_end = int(len(data.index) * ((fold_num + 1) / n_folds))
        # The total number of rows that are for testing
        test_total = test_end - test_start
        # The range of indices for the testing rows
        test_range = range(test_start, test_end)
        #print("Length of Data: " + str(len(data.index)) +
        #      "\nn_folds: " + str(n_folds) +
        #      "\nfold_num: " + str(fold_num) +
        #      "\ntest_start: " + str(test_start) +
        #      "\ntest_end: " + str(test_end) +
        #      "\ntest_range: " + str(test_range))

        # Split off the testing data
        test = data.iloc[test_start:test_end]
        # Copy the labels from the testing data
        test_label_df = test.iloc[:, label_col]
        # Delete the labels from the test DataFrame so that only features are now present
        test_feature_df = test.drop(label_col, axis=1)
        # Drop all the testing data from the data DataFrame so that only training data is present
        train = data.drop(test_range)
        # Copy the labels from the training data
        train_label_df = train.iloc[:, label_col]
        # Delete the labels from the train DataFrame so that only the features are now present
        train_feature_df = train.drop(label_col, axis=1)

        #print("test_feature_df.shape:  " + str(test_feature_df.shape))
        #print("test_label_df.shape:    " + str(test_label_df.shape))
        #print("train_feature_df.shape: " + str(train_feature_df.shape))
        #print("train_label_df.shape:   " + str(train_label_df.shape))

        # Keep track of the number of correct predictions so that the accuracy can be calculated later
        num_success = 0
        # Iterate over the testing data's features and labels
        for (test_num, test_feature), (_, test_label) in zip(test_feature_df.iterrows(), test_label_df.iteritems()):
            # Print the progress
            # print(str(test_num - test_start) + "/" + str(test_total))
            # Create a list to store the k nearest neighbors
            nearest_neighbors = list()
            # Iterate over the training data's features and labels and determine if each one is a closer neighbor than
            # the last
            for (_, train_feature), (_, train_label) in zip(train_feature_df.iterrows(), train_label_df.iteritems()):
                # Cast the Dataframes to 2D arrays
                test_feature_array = [test_feature.values.tolist()]
                train_feature_array = [train_feature.values.tolist()]
                # Calculate the distance between the two using the euclidean distance
                dist = spatial.distance.cdist(test_feature_array, train_feature_array, metric='euclidean')
                #print("distance = " + str(dist))
                # If the list of nearest neighbors is not full yet, just go ahead and add it
                if len(nearest_neighbors) < k_neighbors:
                    #print("Adding the " + str(len(nearest_neighbors)) + " few: " + str(train_label) + ", dist: " + str(dist))
                    nearest_neighbors.append(Neighbor(train_label, dist))
                else:
                    # If it's full, go through the list of current nearest neighbors
                    for index, neighbor in enumerate(nearest_neighbors):
                        #print("dist: " + str(dist))
                        #print("index: " + str(index))
                        #print("neighbor[1]: " + str(neighbor[1]))

                        # Check if the distance is shorter
                        if dist < neighbor.dist:
                            #print("Found nearer neighbor: " + str(train_label) + ", dist: " + str(dist))
                            # If the distance is shorter, replace the further neighbor with the current one
                            nearest_neighbors[index] = Neighbor(train_label, dist)

            # Create a list of Tallies that are used to determine the most number of instances of a certain label among
            # the nearest neighbors
            tallies = list()
            # Iterate over the k nearest neighbors
            for neighbor in nearest_neighbors:
                # Check if the vote has been counted yet
                added = False
                # Iterate through the current tallies to see if there exists one for the current neighbor's label
                for tally in tallies:
                    # If the tally is for the neighbor's label
                    if neighbor.label == tally.label:
                        # Increase the number of occurrences of the current neighbor's label
                        tally.count += 1
                        # Signal that the neighbor has been added to its corresponding tally
                        added = True
                # If there exists no tally that corresponds to the current neighbor's label
                if not added:
                    # Add another Tally to the list of Tallies with 1 vote
                    tallies.append(Tally(neighbor.label, 1))

            # Create a new Tally Object that will hold the most voted on tally
            most_count_tally = None
            # Iterate over all the tallies
            for tally in tallies:
                # If this is the first tally
                if most_count_tally == None:
                    # Make the current tally the most voted
                    most_count_tally = tally
                # If the current tally has more votes than the previous most voted on tally
                elif tally.count > most_count_tally.count:
                    # Make the current tally the new most voted
                    most_count_tally = tally
            # Compare the predicted label
            # (the label from the tally with the most votes)
            # (The label with the most occurrences within the nearest neighbors)
            # to the actual/original label
            if most_count_tally.label == test_label:
                # If the prediction is correct, increment the number of successful predictions
                num_success += 1

        # The accuracy of the current fold can be calculated by taking the number of successful predictions and dividing
        # it by the total number of predictions made
        accuracy = num_success / test_total
        print("Fold #" + str(fold_num) + " accuracy: " + str(accuracy))
        # Append this fold's accuracy to the list of all the folds' accuracies
        accuracies.append(accuracy)
    # The non-averaged sum of all accuracies
    total_accuracy = 0
    # Iterate over all the folds' accuracies
    for accuracy in accuracies:
        # Add them together
        total_accuracy += accuracy
    # Average them
    total_accuracy /= n_folds
    return total_accuracy


##
# Runs the thing
# @param filename String The name of the file to read from that must be located in the same directory
# @param delimit String That is the delimiter for the items in a row
# @param label_col int Which column (starting from 0) is the one with the label
# @param remove_col array[int] The columns to remove
# @param header bool Whether or not there is a header
# @return float The accuracy of the k-NN prediction algorithm
#
def thing(filename, delimit, label_col, remove_col, header, n_folds, k_neighbors):
    # A Pandas DataFrame with all the data. Both training and testing
    data = pd.DataFrame()
    # Removes the first row if the columns are labeled
    if header:
        data = pd.read_csv(filename, delimiter=delimit, index_col=False)
    else:
        data = pd.read_csv(filename, delimiter=delimit, header=None, index_col=False)
    # Removes any rows that are not wanted
    data = data.drop(remove_col, axis=1)

    for col_num in remove_col:
        if label_col > col_num:
            label_col -= 1
        elif label_col == col_num:
            print("Cannot Delete the Label Column")

    # Print the first row to make sure the data is good
    #print(data.iloc[1])

    # Print the shape to see how much data we have
    # print(data.shape)

    # Run the algorithm
    test_accuracy = test(data, label_col, n_folds, k_neighbors)
    return test_accuracy


breast_accuracy = thing("breast-cancer-wisconsin.data", ",", 10, [0], False, 5, 5)
print("Breast Accuracy: " + str(breast_accuracy))
test_accuracy = thing("train-file_1", ' ', 0, [3], False, 5, 5)
print("Test Accuracy: " + str(test_accuracy))

