import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
tf.random.set_seed(777)  # for reproducibility

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers

from pso import Optimizer

import gc

def make_model():
    model = Sequential()
    model.add(layers.Dense(10, activation='relu', input_shape=(4,)))
    model.add(layers.Dense(10, activation='relu'))
    model.add(layers.Dense(3, activation='softmax'))

    return model

def load_data():
    iris = load_iris()
    x = iris.data
    y = iris.target

    y = keras.utils.to_categorical(y, 3)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True, stratify=y)

    return x_train, x_test, y_train, y_test

model = make_model()
x_train, x_test, y_train, y_test = load_data()

loss = 'categorical_crossentropy'

pso_iris = Optimizer(model, loss=loss, n_particles=50, c0=0.5, c1=0.8, w_min=0.7, w_max=1.3)

weight, score = pso_iris.fit(
    x_train, y_train, epochs=500, save=True, save_path="./result/iris", renewal="acc", empirical_balance=False, Dispersion=False, check_point=50)

pso_iris.model_save("./result/iris")
pso_iris.save_info("./result/iris/")

gc.collect()
