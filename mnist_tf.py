# %%
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices("GPU")
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except Exception as e:
        print(e)
    finally:
        del gpus

from keras.datasets import mnist
from keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from keras.models import Sequential

from pso import Optimizer


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


model = make_model()
x_train, y_train, x_test, y_test = get_data()

y_train = tf.one_hot(y_train, 10)
y_test = tf.one_hot(y_test, 10)

model.compile(
    optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
)

# model.compile(optimizer="adam", loss="mse", metrics=["accuracy"])

print("Training model...")
model.fit(x_train, y_train, epochs=100, batch_size=128, verbose=1)

print("Evaluating model...")
model.evaluate(x_test, y_test, verbose=1)

weights = model.get_weights()

for w in weights:
    print(w.shape)
    print(w)
    print(w.min(), w.max())

model.save_weights("weights.h5")

# %%
for w in weights:
    print(w.shape)
    print(w)
    print(w.min(), w.max())

# %%
