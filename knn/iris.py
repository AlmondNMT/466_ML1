import matplotlib.pyplot as plt
import numpy as np
import random
from util import *

"""
Iris CSV Structure
sepal length, sepal width, petal length, petal width
"""
random.seed(1)

def get_iris_numbers(iris_labels):
    iris_labels = list(set(iris_labels))
    iris_map = dict(zip([i for i in range(len(iris_labels))], iris_labels))

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
    attrs, labels = read_iris()
    labels, encoder = encode_labels(labels)
    train_attrs, train_labels, test_attrs, test_labels = get_train_test_split(attrs, labels, 0.5)
    train_avgs = get_averages(train_attrs, train_labels) # Problem 6
    test_avgs = get_averages(test_attrs, test_labels) # Problem 6
    ks = []
    for k in range(1, 2 + 1):
        ks.append(k)
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(train_attrs)
        centroids = []
        for j in range(0, k):
            label_indices = np.where(kmeans.labels_ == j)[0]
            cluster_label = get_mode(train_labels[label_indices])
            centroids.append((kmeans.cluster_centers_[j], cluster_label))
