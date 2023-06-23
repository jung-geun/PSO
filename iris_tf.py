import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices("GPU")
if gpus:
    try:
        # tf.config.experimental.set_visible_devices(gpus[0], "GPU")
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e:
        print(e)
        
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


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

if __name__ == "__main__":
    model = make_model()
    x_train, x_test, y_train, y_test = load_data()
    print(x_train.shape, y_train.shape)

    loss = ['categorical_crossentropy', 'accuracy','mse']
    metrics = ['accuracy']
    
    model.compile(optimizer='sgd', loss=loss[0], metrics=metrics[0])
    model.fit(x_train, y_train, epochs=200, batch_size=32, validation_split=0.2)
    model.evaluate(x_test, y_test, batch_size=32)  