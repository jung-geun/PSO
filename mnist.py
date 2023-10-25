# %%
from pso import optimizer
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from keras.datasets import mnist
import tensorflow as tf
import numpy as np
import json
import os
import sys

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


def get_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train, x_test = x_train / 255.0, x_test / 255.0
    x_train = x_train.reshape((60000, 28, 28, 1))
    x_test = x_test.reshape((10000, 28, 28, 1))

    y_train, y_test = tf.one_hot(y_train, 10), tf.one_hot(y_test, 10)

    x_train, x_test = tf.convert_to_tensor(
        x_train), tf.convert_to_tensor(x_test)
    y_train, y_test = tf.convert_to_tensor(
        y_train), tf.convert_to_tensor(y_test)

    print(f"x_train : {x_train[0].shape} | y_train : {y_train[0].shape}")
    print(f"x_test : {x_test[0].shape} | y_test : {y_test[0].shape}")

    return x_train, y_train, x_test, y_test


def make_model():
    model = Sequential()
    model.add(
        Conv2D(32, kernel_size=(5, 5), activation="relu",
               input_shape=(28, 28, 1))
    )
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))
    model.add(Conv2D(64, kernel_size=(3, 3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(256, activation="relu"))
    model.add(Dense(128, activation="relu"))
    model.add(Dense(10, activation="softmax"))

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
    c1=0.8,
    w_min=0.6,
    w_max=0.9,
    negative_swarm=0.0,
    mutation_swarm=0.2,
    convergence_reset=True,
    convergence_reset_patience=10,
    convergence_reset_monitor="loss",
    convergence_reset_min_delta=0.05,
)

best_score = pso_mnist.fit(
    x_train,
    y_train,
    epochs=300,
    save_info=True,
    log=2,
    log_name="mnist",
    renewal="loss",
    check_point=25,
    empirical_balance=True,
    dispersion=False,
    batch_size=2048,
    back_propagation=False,
    validate_data=(x_test, y_test),
)

print("Done!")

sys.exit(0)
