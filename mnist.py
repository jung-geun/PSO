# %%
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import gc

import tensorflow as tf
from keras import backend as K
from keras.datasets import mnist
from keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from keras.models import Sequential
from pso import Optimizer
from tensorflow import keras
from tqdm import tqdm


def get_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train, x_test = x_train / 255.0, x_test / 255.0
    x_train = x_train.reshape((60000, 28, 28, 1))
    x_test = x_test.reshape((10000, 28, 28, 1))

    print(f"x_train : {x_train[0].shape} | y_train : {y_train[0].shape}")
    print(f"x_test : {x_test[0].shape} | y_test : {y_test[0].shape}")
    return x_train, y_train, x_test, y_test


def get_data_test():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_test = x_test.reshape((10000, 28, 28, 1))

    return x_test, y_test


def make_model():
    model = Sequential()
    model.add(
        Conv2D(32, kernel_size=(5, 5), activation="relu", input_shape=(28, 28, 1))
    )
    model.add(MaxPooling2D(pool_size=(3, 3)))
    model.add(Conv2D(64, kernel_size=(3, 3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation="relu"))
    model.add(Dense(10, activation="softmax"))

    return model


# %%
model = make_model()
x_test, y_test = get_data_test()

loss = [
    "mse",
    "categorical_crossentropy",
    "binary_crossentropy",
    "kullback_leibler_divergence",
    "poisson",
    "cosine_similarity",
    "log_cosh",
    "huber_loss",
    "mean_absolute_error",
    "mean_absolute_percentage_error",
]

if __name__ == "__main__":
    try:
        pso_mnist = Optimizer(
            model,
            loss=loss[0],
            n_particles=100,
            c0=0.35,
            c1=0.8,
            w_min=0.7,
            w_max=1.1,
            negative_swarm=0.2,
            mutation_swarm=0.1,
        )

        best_score = pso_mnist.fit(
            x_test,
            y_test,
            epochs=200,
            save=True,
            save_path="./result/mnist",
            renewal="acc",
            empirical_balance=False,
            Dispersion=False,
            check_point=25,
        )
    except Exception as e:
        print(e)
    finally:
        gc.collect()
