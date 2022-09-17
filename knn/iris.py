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
    attrs, labels = read_iris()
    train_attrs, train_labels, test_attrs, test_labels = get_train_test_split(attrs, labels, 0.5)
    train_avgs = get_averages(train_attrs, train_labels) # Problem 6
    test_avgs = get_averages(test_attrs, test_labels) # Problem 6
    ks = []
    train_acc = []
    test_acc = []
    for k in range(1, 15 + 1):
        ks.append(k)
        kmeans_train = KMeans(n_clusters=k)
        kmeans_train.fit(train_attrs)
        kmeans_test = KMeans(n_clusters=k)
        kmeans_test.fit(test_attrs)
        centroids_train = []
        centroids_test = []
        for j in range(0, k):
            label_indices = np.where(kmeans_train.labels_ == j)[0]
            cluster_label = get_mode(train_labels[label_indices])
            centroids_train.append((kmeans_train.cluster_centers_[j], cluster_label))
            label_indices = np.where(kmeans_test.labels_ == j)[0]
            cluster_label = get_mode(test_labels[label_indices])
            centroids_test.append((kmeans_test.cluster_centers_[j], cluster_label))
            
        train_pred = predict_by_centroids(centroids_train, train_attrs, cosine, False)
        test_pred = predict_by_centroids(centroids_test, test_attrs, cosine, False)
        train_acc.append(get_accuracy(train_pred, train_labels))
        test_acc.append(get_accuracy(test_pred, test_labels))
    
    fig, ax = plt.subplots()
    ax.plot(ks, train_acc, color="r")
    ax.plot(ks, test_acc, color="b")
    fig.savefig("images/elbow_map_kmeans_iris_cosine.png")
