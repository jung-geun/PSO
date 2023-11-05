import os
import sys

import pandas as pd
import tensorflow as tf
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical


gpus = tf.config.experimental.list_physical_devices("GPU")
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as r:
        print(r)

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


def make_model():
    model = Sequential()
    model.add(Dense(12, input_dim=64, activation="relu"))
    model.add(Dense(12, activation="relu"))
    model.add(Dense(10, activation="softmax"))

    return model


def get_data():
    digits = load_digits()
    X = digits.data
    y = digits.target

    x = X.astype("float32")

    y_class = to_categorical(y)

    x_train, x_test, y_train, y_test = train_test_split(
        x, y_class, test_size=0.2, random_state=42, shuffle=True
    )
    return x_train, x_test, y_train, y_test


if __name__ == "__main__":
    model = make_model()
    x_train, x_test, y_train, y_test = get_data()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=10, restore_best_weights=True
        )
    ]

    print(x_train.shape, y_train.shape)

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy", "mse"],
    )

    print(model.summary())

    history = model.fit(
        x_train,
        y_train,
        epochs=500,
        batch_size=32,
        verbose=1,
        validation_data=(x_test, y_test),
        callbacks=callbacks,
    )

    print("Done!")

    sys.exit(0)
