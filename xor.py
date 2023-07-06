# %%
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import tensorflow as tf

# from pso_tf import PSO
from pso import Optimizer
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

print(tf.__version__)
print(tf.config.list_physical_devices())


def get_data():
    x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])
    return x, y


def make_model():
    leyer = []
    leyer.append(layers.Dense(2, activation="sigmoid", input_shape=(2,)))
    # leyer.append(layers.Dense(2, activation='sigmoid'))
    leyer.append(layers.Dense(1, activation="sigmoid"))

    model = Sequential(leyer)

    return model


# %%
model = make_model()
x_test, y_test = get_data()

loss = [
    "mean_squared_error",
    "mean_squared_logarithmic_error",
    "binary_crossentropy",
    "categorical_crossentropy",
    "sparse_categorical_crossentropy",
    "kullback_leibler_divergence",
    "poisson",
    "cosine_similarity",
    "log_cosh",
    "huber_loss",
    "mean_absolute_error",
    "mean_absolute_percentage_error",
]

pso_xor = Optimizer(
    model,
    loss=loss[0],
    n_particles=75,
    c0=0.35,
    c1=0.8,
    w_min=0.6,
    w_max=1.2,
    negative_swarm=0.25,
    mutation_swarm=0.25,
)
best_score = pso_xor.fit(
    x_test,
    y_test,
    epochs=200,
    save=True,
    save_path="./result/xor",
    renewal="acc",
    empirical_balance=False,
    Dispersion=False,
    check_point=25,
)

# %%
