"""
Functions for retrieving the images and labels from the MNIST binary files dataset.
"""

import json
import numpy as np
import os
from PIL import Image
import random
import time
from util import *

# Image training data files
directory = "data/"
training_images_filename = "train-images-idx3-ubyte"
training_labels_filename = "train-labels-idx1-ubyte"
testing_images_filename = "t10k-images-idx3-ubyte"
testing_labels_filename = "t10k-labels-idx1-ubyte"

def bytes_to_int(byte_data):
    """
    Big Endian byte storage conversion from bytecode to decimal integer 
    A wrapper for int.from_bytes(bytes, byteorder = "big", signed = False) 
    """
    return int.from_bytes(byte_data, byteorder = "big", signed = False)


def get_images_array(image_filename):
    """
    This takes the bytes of the ubyte image file and first finds the total 
    number of images to extract. 
    """
    image_bytes = read_bytes(image_filename)
    images = []
    num_images = bytes_to_int(image_bytes[4 : 8])
    rows = bytes_to_int(image_bytes[8 : 12])
    cols = bytes_to_int(image_bytes[12 : 16])
    for i in range(num_images):
        images.append(np.array(bytearray(image_bytes[16 + i * rows * cols : 16 + rows * cols * (i + 1)])) / (255) )
    return images

def get_training_and_validation(training_count = 50000):
    """
    Determine how to split the training and validation images 
    """
    if 0 >= training_count or training_count >= 60000:
        raise ValueError("Training image count must be between 0 and 60000.")
    images = get_images_array(directory + training_images_filename)
    labels = get_labels(directory + training_labels_filename)
    training_images = images[0 : training_count] 
    training_labels = labels[0 : training_count]
    validation_images = images[training_count : 60000]
    validation_labels = labels[training_count : 60000]
    return training_images, training_labels, validation_images, validation_labels

def get_training_and_testing(split = 0.5):
    assert 0.1 <= split <= .9, "split must be between 0 and 1"
    images = get_images_array(directory + training_images_filename)
    labels = get_labels(directory + training_labels_filename)
    test_images = get_images_array(directory + testing_images_filename)
    test_labels = get_labels(directory + testing_labels_filename)
    images += test_images
    labels += test_labels
    images_and_labels = list(zip(images, labels))
    random.shuffle(images_and_labels) # Randomize the images and labels
    images, labels = zip(*images_and_labels) 
    images = np.array(images)
    labels = np.array(labels)
    n = round(len(images) * split)
    training_images, training_labels = images[:n], labels[:n]
    testing_images, testing_labels = images[n:], labels[n:]
    return training_images, training_labels, testing_images, testing_labels

# Get the labels corresponding to the images 
def get_labels(label_filename):
    label_bytes = read_bytes(label_filename)
    num_labels = bytes_to_int(label_bytes[4 : 8])
    labels = [[0 for i in range(10)] for j in range(num_labels)]
    for i in range(num_labels):
        labels[i][label_bytes[8 + i]] = 1
    return labels

def get_testing_images():
    return (get_images_array(directory + testing_images_filename), 
            get_labels(directory + testing_labels_filename))

def read_bytes(filename):
    with open(filename, "rb") as f:
        file_bytes = f.read()
        return file_bytes

def organize_labels(images, labels):
    assert len(images) == len(labels), "images and labels must be the same size"
    return sorted(zip(images, labels), key=lambda t: np.argmax(t[1]))

def get_image(array):
    """
    pass a flat image pixel array and display the image
    """
    assert type(array) is np.ndarray, "array must be a numpy array"
    img = Image.fromarray(np.array((255 * array).reshape(28, 28), 
        dtype=np.uint8), mode="L")
    return img

def get_partition_indices(zipped_sorted):
    """
    This takes a sorted data set zipped with the corresponding labels and 
    finds the indices bounding each partition
    """
    prev = 0
    indices = dict()
    for i, (image, one_hot) in enumerate(zipped_sorted):
        label = np.argmax(one_hot)
        assert label >= prev, "data needs to be sorted"
        if not indices.get(label):
            indices[label] = [i]
        if label > prev:
            prev = label
            indices[label - 1].append(i)
    indices[label].append(i)
    return indices

def get_closest_to_average(images, labels):
    """
    :return: the images closest to the averages using euclidean, manhattan, and
    cosine distance metrics.
    """
    averages = get_averages(images, labels)
    closest = []
    start = time.time()
    for label in averages:
        avg_pixels = averages[label][0]
        eucl_min = images[np.argmin(euclidean(avg_pixels, images))]
        manh_min = images[np.argmin(manhattan(avg_pixels, images))]
        cos_min = images[np.argmax(cosine(avg_pixels, images))]
        closest.extend([(label, (eucl_min, manh_min, cos_min))])
    print("Closest to average calculation duration: %.2f s" % (time.time() - start))
    return closest

def save_averages(avg_dict, partition_name: str) -> None:
    """
    :param avg_dict: dictionary of labels and their averages
    :param partition_name: a string for the filename suffix
    :return: None
    """
    for label in avg_dict:
        img = get_image(avg_dict[label][0])
        img.save("images/{}_{}_avg.png".format(label, partition_name))

def save_closest(closest, partition_name: str):
    for item in closest:
        label, triplet = item
        for typ, val in zip(["euclidean", "manhattan", "cosine"], triplet):
            img = get_image(val)
            img.save("images/{}_{}_{}_min.png".format(label, partition_name, typ))

if __name__ == "__main__":
    training_images, training_labels, testing_images, testing_labels = get_training_and_testing()
    training_averages = get_averages(training_images, training_labels)
    testing_averages = get_averages(testing_images, testing_labels)
    save_averages(training_averages, "train")
    save_averages(testing_averages, "test")
