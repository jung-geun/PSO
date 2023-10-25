import os
import sys

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import gc

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

from pso import optimizer


def make_model():
    model = Sequential()
    model.add(layers.Dense(10, activation="relu", input_shape=(4,)))
    model.add(layers.Dense(10, activation="relu"))
    model.add(layers.Dense(3, activation="softmax"))

    return model


def load_data():
    iris = load_iris()
    x = iris.data
    y = iris.target

    y = keras.utils.to_categorical(y, 3)

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, shuffle=True, stratify=y
    )

    return x_train, x_test, y_train, y_test


model = make_model()
x_train, x_test, y_train, y_test = load_data()

loss = ["categorical_crossentropy", "mean_squared_error"]

pso_iris = optimizer(
    model,
    loss=loss[1],
    n_particles=100,
    c0=0.35,
    c1=0.6,
    w_min=0.5,
    w_max=0.9,
    negative_swarm=0,
    mutation_swarm=0.2,
    convergence_reset=True,
    convergence_reset_patience=10,
    convergence_reset_monitor="mse",
    convergence_reset_min_delta=0.05,
)

best_score = pso_iris.fit(
    x_train,
    y_train,
    epochs=500,
    save_info=True,
    log=2,
    log_name="iris",
    renewal="acc",
    check_point=25,
    validate_data=(x_test, y_test),
)

gc.collect()
print("Done!")
sys.exit(0)
