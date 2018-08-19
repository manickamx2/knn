# Manickam Manickam
# knn.py - k-nearest-neighbor algorithm implementation
# 8.7.2018

import sys
import math
import random
import instance as inst
from collections import Counter
from itertools import zip_longest

##### SPLIT VALUES INTO N-SIZE CHUNKS #####
def grouper(iterable, n, fillValue=None):
    args = [iter(iterable)] * n
    return zip_longest(*args, fillvalue=fillValue)

##### PREPROCESS THE DATA #####
def preprocessData(data_file, instanceList):
    possible_labels = []
    next(data_file)  # skip the first line of the data file: ['label', attribute_1, attribute_2, ..., attribute_n]
    for line in data_file:
        arr = line.split(",")
        instance = inst.Instance(arr)
        instance.process()
        instanceList.append(instance)
        if instance.getLabel() not in possible_labels:
            possible_labels.append(instance.getLabel())
    return possible_labels

##### SEPARATE INTO TRAINING AND TEST SETS #####
def partitionData(data_set, training_percentage):
    training_length = math.ceil(len(data_set) * training_percentage)
    count = 0
    training_set = []
    test_set = []
    for i in range(len(data_set)):
        if count < training_length:
            training_set.append(data_set[i])
            count += 1
        else:
            test_set.append(data_set[i])

    return training_set, test_set

##### MAKE PREDICTION BASED ON K-MOST SIMILAR INSTANCES FROM TRAINING SET IN COMPARISON TO TEST INSTANCE
def distanceFunction(distance_func, test_instance, training_set, k):
    smallest_distances = [] #list of k instances from training set w/ smallest distance. has length of k.
    if distance_func == "H": #Hamming Distance, nominal data.
        test_instance_attributes = test_instance.getAttributes() #the test instance's set of attributes.
        for training_instance in training_set:
            training_instance_attributes = training_instance.getAttributes() #set of attributes for an instance of the training set.
            distance = 0 #preliminarily set distance to 0.
            for i in range(len(training_instance_attributes)): #loop through attributes of training set and compare to those of test set.
                if test_instance_attributes[i] != training_instance_attributes[i]: #when attributes are not equivalent...
                    distance += 1 #increase the distance by 1.
            smallest_distances.append((distance, training_instance))
        #smallest_distances[0][0] contains the distance values we want to sort by.
        smallest_distances = sorted(smallest_distances, key = lambda x: x[0]) #sort by distance.
        smallest_distances = smallest_distances[0:k] #shorten list to length of k.

    if distance_func == "E": #Euclidean Distance, continuous data.
        test_instance_attributes = test_instance.getAttributes()  #the test instance's set of attributes.
        for training_instance in training_set:
            training_instance_attributes = training_instance.getAttributes() #the training instance's set of attributes.
            s = 0
            for i in range(len(training_instance_attributes)):
                s += (test_instance_attributes[i] - training_instance_attributes[i])**2 #sum of squares of difference between test attribute and training attribute.
            distance = math.sqrt(s)
            smallest_distances.append((distance, training_instance))
        smallest_distances = sorted(smallest_distances, key = lambda x: x[0]) #sort by distance.
        smallest_distances = smallest_distances[0:k] #shorten list to length of k.

    #collect the most similar labels from smallest_distances
    #https://stackoverflow.com/questions/47843707/count-frequency-of-item-in-a-list-of-tuples.
    #count number of times particular label occurs, given by one-liner from SO.
    label_counts = Counter(x[1].getLabel() for x in smallest_distances)
    max_label = label_counts.most_common(1)[0][0] #the label that occurs the most from the count of labels.

    return max_label

##### GENERATE CONFUSION MATRIX #####
def confusionMatrix(correct_predictions, testSet, possible_labels, confusion_matrix):
    print()
    print("########### CONFUSION MATRIX ##########")
    print()

    print("Accuracy: ", float(correct_predictions) / len(testSet))

    for predicted in possible_labels:
        predicted = predicted.strip('\"')
        predicted = predicted + ","
        print(predicted, end="")
    print()
    values = list(confusion_matrix.values())
    index = 0
    for group in grouper(values, len(possible_labels)):
        for value in group:
            value = str(value) + ","
            print(value, end="")
        actual = possible_labels[index].strip('\"')
        print(actual)
        index += 1
    print()
    print("########### CONFUSION MATRIX ##########")
    print()


def main():

    #parse arguments
    data_file = sys.argv[1]
    distance_func = sys.argv[2]
    k = int(sys.argv[3])
    training_percentage = float(sys.argv[4])
    seed = int(sys.argv[5])

    instanceList = []
    correct_predictions = 0
    labels = []
    confusion_matrix = {}

    #1) open read in data as set of instances
    data_file = open(data_file, "r")
    possible_labels = preprocessData(data_file, instanceList)
    data_file.close()

    #2) shuffle the list
    random.seed(seed)
    shuffledInstances = list(instanceList)
    random.shuffle(shuffledInstances)

    #3) separate into training and test sets
    trainingSet, testSet = partitionData(shuffledInstances, training_percentage)


    #4) make predictions using the training set as instances to compare to in k-nearest-neighbor
    for actual in possible_labels:
        for predicted in possible_labels:
            correct = 0
            label_pair = actual, predicted
            for testInstance in testSet:
                prediction = distanceFunction(distance_func, testInstance, trainingSet, k)
                #did the predicted label match the actual label?
                if testInstance.getLabel() == prediction:
                    correct += 1

                #need to count actual/predicted pairs.
                if label_pair not in confusion_matrix:
                    confusion_matrix[label_pair] = 0
                    if actual == testInstance.getLabel() and predicted == prediction:
                        confusion_matrix[label_pair] += 1
                else:
                    if actual == testInstance.getLabel() and predicted == prediction:
                        confusion_matrix[label_pair] += 1
    correct_predictions += correct


    #5) produce the confusion matrix as output in readable format.
    confusionMatrix(correct_predictions, testSet, possible_labels, confusion_matrix)

main()