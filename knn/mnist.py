"""
Functions for retrieving the images and labels from the MNIST binary files dataset.
"""

import json
import numpy as np
import os
from PIL import Image
import random

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
    assert 0 < split < 1, "split must be between 0 and 1"
    images = get_images_array(directory + training_images_filename)
    labels = get_labels(directory + training_labels_filename)
    test_images = get_images_array(directory + testing_images_filename)
    test_labels = get_labels(directory + testing_labels_filename)
    images += test_images
    labels += test_labels
    images_and_labels = list(zip(images, labels))
    random.shuffle(images_and_labels) # Randomize the images and labels
    images, labels = zip(*images_and_labels) 
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

def get_averages(images, labels):
    """
    Determine the average templates for each label
    """
    averages = dict()
    for pixels, one_hot in zip(images, labels):
        label = np.argmax(one_hot)
        if not averages.get(label):
            averages[label] = (pixels, 1)
            continue
        avg, count = averages[label]
        averages[label] = ((avg * count + pixels) / (count + 1), count + 1)
    return averages
def get_image(array):
    """
    pass a flat image pixel array and display the image
    """
    assert type(array) is np.ndarray, "array must be a numpy array"
    img = Image.fromarray(np.array((255 * array).reshape(28, 28), 
        dtype=np.uint8), mode="L")
    return img
def save_averages(avg_dict, partition_name: str) -> None:
    """
    :param avg_dict: dictionary of labels and their averages
    :param partition_name: a string for the filename suffix
    :return: None
    """
    for label in avg_dict:
        img = get_image(avg_dict[label][0])
        img.save("data/{}_{}_avg.png".format(label, partition_name))
if __name__ == "__main__":
    training_images, training_labels, testing_images, testing_labels = get_training_and_testing()
    training_averages = get_averages(training_images, training_labels)
    testing_averages = get_averages(testing_images, testing_labels)
    save_averages(training_averages, "train")
    save_averages(testing_averages, "test")

    img = get_image(training_images[0])
