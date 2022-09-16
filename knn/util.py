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
