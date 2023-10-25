# %%
import json
import os
import sys

import numpy as np
import pandas as pd
import tensorflow as tf
from keras.layers import Dense
from keras.models import Sequential
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow import keras

from pso import optimizer

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


def get_data():
    with open("data/seeds/seeds_dataset.txt", "r", encoding="utf-8") as f:
        data = f.readlines()
        df = pd.DataFrame([d.split() for d in data])
        df.columns = [
            "area",
            "perimeter",
            "compactness",
            "length_of_kernel",
            "width_of_kernel",
            "asymmetry_coefficient",
            "length_of_kernel_groove",
            "target",
        ]

        df = df.astype(float)
        df["target"] = df["target"].astype(int)

        x = df.iloc[:, :-1].values.round(0).astype(int)
        y = df.iloc[:, -1].values

    y_class = to_categorical(y)

    x_train, x_test, y_train, y_test = train_test_split(
        x, y_class, test_size=0.2, shuffle=True
    )

    return x_train, y_train, x_test, y_test


def make_model():
    model = Sequential()
    model.add(Dense(16, activation="relu", input_shape=(7,)))
    model.add(Dense(32, activation="relu"))
    model.add(Dense(4, activation="softmax"))

    return model


# %%
model = make_model()
x_train, y_train, x_test, y_test = get_data()

loss = [
    "mean_squared_error",
    "categorical_crossentropy",
    "sparse_categorical_crossentropy",
    "binary_crossentropy",
    "kullback_leibler_divergence",
    "poisson",
    "cosine_similarity",
    "log_cosh",
    "huber_loss",
    "mean_absolute_error",
    "mean_absolute_percentage_error",
]

# rs = random_state()

pso_mnist = optimizer(
    model,
    loss="categorical_crossentropy",
    n_particles=100,
    c0=0.5,
    c1=1.0,
    w_min=0.7,
    w_max=1.2,
    negative_swarm=0.0,
    mutation_swarm=0.3,
    convergence_reset=True,
    convergence_reset_patience=10,
    convergence_reset_monitor="mse",
    convergence_reset_min_delta=0.0005,
)

best_score = pso_mnist.fit(
    x_train,
    y_train,
    epochs=500,
    save_info=True,
    log=2,
    log_name="seeds",
    renewal="acc",
    check_point=25,
    empirical_balance=False,
    dispersion=False,
    back_propagation=False,
    validate_data=(x_test, y_test),
)

print("Done!")

sys.exit(0)
