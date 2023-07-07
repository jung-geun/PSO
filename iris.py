import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import gc

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

from pso import Optimizer


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

loss = ["categorical_crossentropy"]

pso_iris = Optimizer(
    model,
    loss=loss[0],
    n_particles=100,
    c0=0.4,
    c1=0.8,
    w_min=0.7,
    w_max=1.0,
    negative_swarm=0.1,
    mutation_swarm=0.2,
)

best_score = pso_iris.fit(
    x_train,
    y_train,
    epochs=200,
    save=True,
    save_path="./result/iris",
    renewal="acc",
    empirical_balance=False,
    Dispersion=False,
    check_point=25,
)

gc.collect()
