# %%
from pso import optimizer
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from keras.datasets import mnist, fashion_mnist
import tensorflow as tf
import numpy as np
import json
import os
import sys

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


def get_data():
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

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


def get_data_test():
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    x_test = x_test / 255.0
    x_test = x_test.reshape((10000, 28, 28, 1))

    y_train, y_test = tf.one_hot(y_train, 10), tf.one_hot(y_test, 10)

    x_train, x_test = tf.convert_to_tensor(
        x_train), tf.convert_to_tensor(x_test)
    y_train, y_test = tf.convert_to_tensor(
        y_train), tf.convert_to_tensor(y_test)

    print(f"x_test : {x_test[0].shape} | y_test : {y_test[0].shape}")

    return x_test, y_test


def make_model():
    model = Sequential()
    model.add(
        Conv2D(32, kernel_size=(5, 5), activation="sigmoid",
               input_shape=(28, 28, 1))
    )
    model.add(MaxPooling2D(pool_size=(3, 3)))
    model.add(Conv2D(64, kernel_size=(3, 3), activation="sigmoid"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation="sigmoid"))
    model.add(Dense(10, activation="softmax"))

    return model


def random_state():
    with open(
        "result/mnist/20230723-061626/mean_squared_error_[0.6384999752044678, 0.0723000094294548].json",
        "r",
    ) as f:
        json_ = json.load(f)
        rs = (
            json_["random_state_0"],
            np.array(json_["random_state_1"]),
            json_["random_state_2"],
            json_["random_state_3"],
            json_["random_state_4"],
        )

    return rs


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
    loss="mean_squared_error",
    n_particles=500,
    c0=0.2,
    c1=0.4,
    w_min=0.3,
    w_max=0.5,
    negative_swarm=0.05,
    mutation_swarm=0.3,
    particle_min=-4,
    particle_max=4,
)

best_score = pso_mnist.fit(
    x_train,
    y_train,
    epochs=200,
    save_info=True,
    log=2,
    log_name="fashion_mnist",
    save_path="./logs/fashion_mnist",
    renewal="acc",
    check_point=25,
    empirical_balance=False,
    dispersion=False,
    batch_size=32,
)

print("Done!")

sys.exit(0)
