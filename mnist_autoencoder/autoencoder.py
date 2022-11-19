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

def find_digits(actual, pred, labels):
    counts = dict(zip([i for i in range(10)], [0 for i in range(10)]))
    actual_images = []
    pred_images = []
    pred_dir = "saved_images/predicted"
    actual_dir = "saved_images/actual"
    while sum(counts.values()) < 10:
        print(sum(counts.values()))
        label = labels[np.random.randint(0, len(labels))]
        index = np.argmax(label)
        if counts[index] < 1:
            counts[index] += 1
            actual_images.append((index, actual[label]))
            pred_images.append(actual[label])
    for i in range(len(actual_images)):
        label, actual_image_data = actual_images[i]
        actual = Image.fromarray(actual_image_data.reshape(28, 28) * 255).convert("L").resize((100, 100), Image.Resampling.BICUBIC)
        actual.save(os.path.join(actual_dir, label + ".png"))
        pred = Image.fromarray(pred_images[i].reshape(28, 28) * 255).convert("L").resize((100, 100), Image.Resampling.BICUBIC)
        pred.save(os.path.join(pred_dir, label + ".png"))

if __name__ == "__main__":
    model = build_model(5)
    if not os.path.isfile("model.pb"):
        model.fit(X_train, X_train, batch_size=40, epochs=2, validation_split=0.2, shuffle=True)
        model.save_model(".")
    else:
        model = tf.keras.models.load_model("model.pb")


    # Model accuracy
    model.evaluate(X_test, y_test)

    # Prediction
    pred = model.predict(X_test)

    print(model.summary())
    find_digits(X_test, pred, y_test)