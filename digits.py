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

from pso import optimizer

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


def make_model():
    model = Sequential()
    model.add(Dense(12, input_dim=64, activation="relu"))
    model.add(Dense(10, activation="relu"))
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


x_train, x_test, y_train, y_test = get_data()
model = make_model()

digits_pso = optimizer(
    model,
    loss="categorical_crossentropy",
    n_particles=300,
    c0=0.5,
    c1=0.3,
    w_min=0.2,
    w_max=0.9,
    negative_swarm=0,
    mutation_swarm=0.1,
    convergence_reset=True,
    convergence_reset_patience=10,
    convergence_reset_monitor="loss",
    convergence_reset_min_delta=0.001,
)

digits_pso.fit(
    x_train,
    y_train,
    epochs=500,
    validate_data=(x_test, y_test),
    log=2,
    save_info=True,
    renewal="loss",
    log_name="digits",
)

print("Done!")

sys.exit(0)
