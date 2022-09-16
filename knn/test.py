<<<<<<< HEAD
import matplotlib.pyplot as plt
import numpy as np
import random
from util import *

"""
Iris CSV Structure
sepal length, sepal width, petal length, petal width
"""
random.seed(1)

def read_csv():
    """
    Pull csv data from data/iris.data
    """
    attributes = []
    labels = []
    with open("data/iris.data", "r") as f:
        lines = f.readlines()
        for line in lines:
            datum = line.split(",")
            label = datum[-1].rstrip()
            attrs = [float(attr) for attr in datum[:-1]]
            attributes.append(attrs)
            labels.append(label)
        return np.array(attributes), np.array(labels)

def get_train_test_split(split: float):
    """
    :param split: float between .1 and .9 which determines 
    how much training data to use 
    """
    assert 0.1 <= split <= 0.9
    attrs, labels = read_csv()
    combined = list(zip(attrs, labels))
    random.shuffle(combined)
    n = round(split * len(combined))
    attrs, labels = zip(*combined)
    train_attrs = np.array(attrs[:n])
    train_labels = np.array(labels[:n])
    test_attrs = np.array(attrs[n:])
    test_labels = np.array(labels[n:])
    return train_attrs, train_labels, test_attrs, test_labels


if __name__ == "__main__":
    train_attrs, train_labels, test_attrs, test_labels = get_train_test_split(0.5)
    train_avgs = get_averages(train_attrs, train_labels) # Problem 6
    test_avgs = get_averages(test_attrs, test_labels) # Problem 6
    ks = []
    
    act =  [0,1,2,0,1,2,0,1,2,2]
    pred = [0,1,1,0,1,1,0,1,2,2]
    print("TESTING GET CONFUSION MATRIX")
    print(get_confusion_matrix(act, pred))

    for k in range(1, 2 + 1):
        ks.append(k)
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(train_attrs)
        centroids = []
        for j in range(0, k):
            label_indices = np.where(kmeans.labels_ == j)[0]
            cluster_label = get_mode(train_labels[label_indices])
            centroids.append((kmeans.cluster_centers_[j], cluster_label))
        
=======
from iris import *
from mnist import *
from util import *
>>>>>>> cf29be6adb4778c3f20ad6dba327a83e89d10008
