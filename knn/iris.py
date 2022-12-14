from xml.dom import pulldom
import matplotlib.pyplot as plt
import numpy as np
import random
from util import *

"""
Iris CSV Structure
sepal length, sepal width, petal length, petal width
"""
random.seed(1)
plt.ion()
def read_iris():
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

if __name__ == "__main__":
    #kmeans_loop(read_iris, euclidean, "iris", "euclidean")
    """train_pred, train_labels, test_pred, test_labels = min_dist(read_iris, cosine, False)
    le = LabelEncoder()
    train_pred = le.fit_transform(train_pred)
    train_labels = le.transform(train_labels)
    test_pred = le.transform(test_pred)
    test_labels = le.transform(test_labels)
    conf_train = get_confusion_matrix(train_pred, train_labels)
    conf_test = get_confusion_matrix(test_pred, test_labels)
    print(conf_train)
    print(conf_test)"""
    knn(read_iris)

