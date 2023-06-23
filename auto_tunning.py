# %%
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
tf.random.set_seed(777)  # for reproducibility

from tensorflow import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

# from pso_tf import PSO
from pso import Optimizer

import numpy as np

from datetime import date
from tqdm import tqdm

import gc

# print(tf.__version__)
# print(tf.config.list_physical_devices())
# print(f"Num GPUs Available: {len(tf.config.list_physical_devices('GPU'))}")

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
    model.add(Conv2D(32, kernel_size=(5, 5),
              activation='relu', input_shape=(28, 28, 1)))
    model.add(MaxPooling2D(pool_size=(3, 3)))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(10, activation='softmax'))

    return model

# %%
model = make_model()
x_test, y_test = get_data_test()
# loss = 'binary_crossentropy'
# loss = 'categorical_crossentropy'
# loss = 'sparse_categorical_crossentropy'
# loss = 'kullback_leibler_divergence'
# loss = 'poisson'
# loss = 'cosine_similarity'
# loss = 'log_cosh'
# loss = 'huber_loss' 
# loss = 'mean_absolute_error'
# loss = 'mean_absolute_percentage_error'
# loss = 'mean_squared_error'

loss = ['mse', 'categorical_crossentropy', 'binary_crossentropy', 'kullback_leibler_divergence', 'poisson', 'cosine_similarity', 'log_cosh', 'huber_loss', 'mean_absolute_error', 'mean_absolute_percentage_error']
n_particles = [50, 75, 100]
c0 = [0.25, 0.35, 0.45, 0.55]
c1 = [0.5, 0.6, 0.7, 0.8, 0.9]
w_min = [0.5, 0.6, 0.7]
w_max = [1.1, 1.2, 1.3]
negative_swarm = [0.25, 0.3, 0.5]
eb = [True, False]
dispersion = [True, False]

if __name__ == "__main__":
    try:
        for loss_ in loss:
            for n in n_particles:
                for c_0 in c0:
                    for c_1 in c1:
                        for w_m in w_min:
                            for w_M in w_max:
                                for n_s in negative_swarm:
                                    pso_mnist = Optimizer(
                                        model,
                                        loss=loss_, 
                                        n_particles=n,
                                        c0=c_0, 
                                        c1=c_1, 
                                        w_min=w_m,
                                        w_max=w_M,
                                        negative_swarm=n_s
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
                                        check_point=25
                                        )
                                    
                                    del pso_mnist
                                    gc.collect() 
                                    tf.keras.backend.clear_session()
                                    
    except KeyboardInterrupt:
        print("KeyboardInterrupt")
    finally:
        print("Finish")
