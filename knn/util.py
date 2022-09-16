from sklearn.cluster import KMeans
import numpy as np

def get_averages(feature_set, labels):
    """
    Determine the average templates for each label
    """
    averages = dict()
    for features, one_hot in zip(feature_set, labels):
        label = np.argmax(one_hot)
        if not averages.get(label):
            averages[label] = (features, 1)
            continue
        avg, count = averages[label]
        averages[label] = ((avg * count + features) / (count + 1), count + 1)
    return averages

def get_mode(array):
    """
    Get element which occurs most in an array
    """
    counts = dict()
    for element in array:
        if not counts.get(element):
            counts[element] = 1
            continue
        counts[element] += 1
    return list(counts.keys())[np.argmax(list(counts.values()))]

def euclidean(u, v):
    assert type(u) is np.ndarray, "var u must be an ndarray"
    assert type(v) is np.ndarray, "var v must be an ndarray"
    return np.linalg.norm(u - v, axis=1)

def manhattan(u, v):
    """
    Sum the absolute difference of the vector components of u and v
    """
    assert type(u) is np.ndarray, "var u must be an ndarray"
    assert type(v) is np.ndarray, "var v must be an ndarray"
    return np.sum(np.abs(u - v), axis=1)

def cosine(u, v):
    assert type(u) is np.ndarray, "var u must be an ndarray"
    assert type(v) is np.ndarray, "var v must be an ndarray"
    if u.shape == v.shape and len(u.shape) == 1:
        return np.dot(u, v) / np.linalg.norm(u) / np.linalg.norm(v)
    elif len(u.shape) == 1 and len(v.shape) == 2 and v.shape[1] == u.shape[0]:
        return np.dot(v, u) / np.linalg.norm(v, axis=1) / np.linalg.norm(u)
    elif len(u.shape) == 2 and len(v.shape) == 1 and v.shape[0] == u.shape[1]:
        return np.dot(u, v) / np.linalg.norm(u, axis=1) / np.linalg.norm(v)
    else:
        return u.dot(v)

def classify_k_means(centroids, attrs, dist, is_min=True):
    """
    :param attrs: feature vectors 
    :param centroids: list of 2-tuples of k-means centroids and their 
        corresponding labels based on voting
    :param dist: distance function
    :param is_min: boolean to determine whether to search for min or max value
        of the given distance metric
    :return: vector of predicted labels for feature vectors
    """
    assert callable(dist), "dist must be a function"
    index = 0 # Index of the minimum distance
    centroid, label = centroids[0]
    distances = []
    centroid_labels = []
    for centroid, label in centroids:
        distances.append(dist(centroid, attrs))
        centroid_labels.append(label)
    distances = np.array(distances)
    centroid_labels = np.array(centroid_labels)
    if is_min:
        return centroid_labels[np.argmin(distances, axis=0)]
    else:
        return centroid_labels[np.argmax(distances, axis=0)]

def get_confusion_matrix(pred, actual):
    assert len(pred) == len(actual), "predicted and actual must be same len"
    conf = dict()
    for i, label in enumerate(pred):
        pass
    

def get_dist_predictions(centroids, attrs, labels):
    eucl_pred = classify_k_means(centroids, attrs, euclidean, True)
    manh_pred = classify_k_means(centroids, attrs, manhattan, True)
    cos_pred = classify_k_means(centroids, attrs, cosine, False)
    eucl_conf = get_confusion_matrix(eucl_pred, labels)
    manh_conf = get_confusion_matrix(eucl_pred, labels)
    cos_conf = get_confusion_matrix(cos_pred, labels)

    return eucl_pred, manh_pred, cos_pred

def get_accuracy(pred, actual):
    assert (len(pred)==len(actual)), "err get_accuracy: passed vectors need same length"
    counter = 0
    for i in range(0, len(pred)):
        if(pred[i]==actual[i]):
            counter = counter + 1

    return counter/len(pred)