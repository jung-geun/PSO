import os

import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from ucimlrepo import fetch_ucirepo

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'


def make_model():
    model = Sequential()
    model.add(Dense(12, input_dim=16, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(7, activation='softmax'))

    return model


def get_data():
    # fetch dataset
    dry_bean_dataset = fetch_ucirepo(id=602)

    # data (as pandas dataframes)
    X = dry_bean_dataset.data.features
    y = dry_bean_dataset.data.targets

    x = X.to_numpy()
    # object to categorical

    x = x.astype('float32')

    y_class = to_categorical(y)

    # metadata
    # print(dry_bean_dataset.metadata)

    # variable information
    # print(dry_bean_dataset.variables)

    # print(X.head())
    # print(y.head())
    # y_class = to_categorical(y)

    x_train, x_test, y_train, y_test = train_test_split(
        x, y_class, test_size=0.2, random_state=42, shuffle=True)
    return x_train, x_test, y_train, y_test


x_train, x_test, y_train, y_test = get_data()
model = make_model()
early_stopping = keras.callbacks.EarlyStopping(
    patience=10, min_delta=0.001, restore_best_weights=True)


model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam', metrics=['accuracy', "mse"])

model.summary()

history = model.fit(x_train, y_train, epochs=150,
                    batch_size=10, callbacks=[early_stopping])
score = model.evaluate(x_test, y_test, verbose=2)
