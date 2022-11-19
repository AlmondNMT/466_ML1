import numpy as np
import os
import pandas as pd
from PIL import Image
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dense, MaxPool2D, Flatten, Reshape

(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1) / 255.0
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1) / 255.0


def build_model(latent_units):
    model = tf.keras.models.Sequential()
    model.add(Conv2D(64, (5, 5), input_shape=(28, 28, 1), activation="relu"))
    model.add(MaxPool2D((2, 2)))
    model.add(Conv2D(60, (5, 5), activation="relu"))
    model.add(Flatten())
    model.add(Dense(latent_units, activation="relu"))
    model.add(Dense(28 * 28, activation="sigmoid"))
    model.add(Reshape((28, 28, 1), input_shape=(28 ** 2,)))
    model.compile(optimizer="adam", metrics=["accuracy"], loss="MSE")
    return model

def find_digits(actual, pred, labels, latent_units):
    counts = dict(zip([i for i in range(10)], [0 for i in range(10)]))
    actual_images = []
    pred_images = []
    root_dir = os.path.join("saved_images", "latent_" + latent_units)
    pred_dir = os.path.join(root_dir, "predicted")
    actual_dir = os.path.join(root_dir, "actual")
    while sum(counts.values()) < 10:
        index = np.random.randint(0, len(labels))
        label = labels[index]
        print(index)
        if counts[label] < 1:
            counts[label] += 1
            actual_images.append((label, actual[index]))
            pred_images.append(pred[index])
    for i in range(len(actual_images)):
        label, actual_image_data = actual_images[i]
        label = str(int(label))
        actual = Image.fromarray(actual_image_data.reshape(28, 28) * 255).convert("L").resize((100, 100), Image.Resampling.BICUBIC)
        actual.save(os.path.join(actual_dir, label + ".png"))
        pred = Image.fromarray(pred_images[i].reshape(28, 28) * 255).convert("L").resize((100, 100), Image.Resampling.BICUBIC)
        pred.save(os.path.join(pred_dir, label + ".png"))

if __name__ == "__main__":
    latent_units = 5
    model = build_model(latent_units)
    f_string = f"latent_{latent_units}"
    if not os.path.isdir(f_string):
        model.fit(X_train, X_train, batch_size=40, epochs=2, validation_split=0.2, shuffle=True)
        model.save(f_string)
    else:
        model = tf.keras.models.load_model(f_string)


    # Model accuracy
    model.evaluate(X_test, X_test)

    # Prediction
    pred = model.predict(X_test)

    print(model.summary())
    find_digits(X_test, pred, y_test)
