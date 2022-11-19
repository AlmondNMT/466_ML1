import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dense, MaxPool2D, Flatten, Reshape

(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1) / 255.0
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1) / 255.0

model = tf.keras.models.Sequential()
model.add(Conv2D(64, (5, 5), input_shape=(28, 28, 1), activation="relu"))
model.add(MaxPool2D((2, 2)))
model.add(Conv2D(60, (5, 5), activation="relu"))
model.add(Flatten())
model.add(Dense(100, activation="relu"))
model.add(Dense(28 * 28, activation="sigmoid"))
model.add(Reshape((28, 28, 1), input_shape=(28 ** 2,)))
model.compile(optimizer="adam", metrics=["accuracy"], loss="MSE")
model.fit(X_train, X_train, batch_size=40, epochs=5, validation_split=0.2, shuffle=True)
model.evaluate(X_test, y_test)

model.evaluate(X_test, X_test)

pred = model.predict(X_test)

print(model.summary())
index = np.random.randint(0, len(pred))
print(y_test[index])
orig_img = Image.fromarray(X_test[index].reshape(28, 28) * 255).convert("L").resize((100, 100), Image.Resampling.BICUBIC)
decoded = Image.fromarray(pred[index].reshape(28, 28) * 255).convert("L").resize((100, 100), Image.Resampling.BICUBIC)
print("Original")
display(orig_img)
print("Decoded")
display(decoded)
