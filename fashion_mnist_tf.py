from keras.models import Sequential
from keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from keras.datasets import mnist, fashion_mnist
from keras.utils import to_categorical
# from tensorflow.data.Dataset import from_tensor_slices
import tensorflow as tf
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


gpus = tf.config.experimental.list_physical_devices("GPU")
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except Exception as e:
        print(e)
    finally:
        del gpus


def get_data():
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

    x_train, x_test = x_train / 255.0, x_test / 255.0
    x_train = x_train.reshape((60000, 28, 28, 1))
    x_test = x_test.reshape((10000, 28, 28, 1))

    print(f"x_train : {x_train[0].shape} | y_train : {y_train[0].shape}")
    print(f"x_test : {x_test[0].shape} | y_test : {y_test[0].shape}")

    return x_train, y_train, x_test, y_test


class _batch_generator:
    def __init__(self, x, y, batch_size: int = 32):
        self.batch_size = batch_size
        self.index = 0
        dataset = tf.data.Dataset.from_tensor_slices((x, y))
        self.dataset = list(dataset.batch(batch_size))
        self.max_index = len(dataset) // batch_size

    def next(self):
        self.index += 1
        if self.index >= self.max_index:
            self.index = 0
        return self.dataset[self.index][0], self.dataset[self.index][1]


# BEGIN: 5f8d9bcejpp
from keras.applications.resnet50 import ResNet50

def make_model():
    model = ResNet50(weights=None, input_shape=(28, 28, 1), classes=10)
    return model
# END: 5f8d9bcejpp


model = make_model()
x_train, y_train, x_test, y_test = get_data()
y_train = tf.one_hot(y_train, 10)
y_test = tf.one_hot(y_test, 10)

dataset = _batch_generator(x_train, y_train, 32)

model.compile(optimizer="adam", loss="mse", metrics=["accuracy"])

count = 0

while count < 50:
    x_batch, y_batch = dataset.next()
    count += 1
    print("Training model...")
    model.fit(x_batch, y_batch, epochs=1, batch_size=1, verbose=1)

print(count)

print("Evaluating model...")
model.evaluate(x_test, y_test, verbose=2)

weights = model.get_weights()
