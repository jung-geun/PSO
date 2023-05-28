from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten

import time as time
import numpy as np


iris = load_iris()
iris = load_iris()
X = iris['data']
y = iris['target']
names = iris['target_names']
feature_names = iris['feature_names']
enc = OneHotEncoder()
Y = enc.fit_transform(y[:, np.newaxis]).toarray()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, Y_train, Y_test = train_test_split(
    X_scaled, Y, test_size=0.5, random_state=2)
n_features = X.shape[1]
n_classes = Y.shape[1]

def create_custom_model(input_dim, output_dim, nodes, n=1, name='model'):
    model = Sequential(name=name)
    for i in range(n):
        model.add(Dense(nodes, input_dim=input_dim, activation='relu'))
    model.add(Dense(output_dim, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model


# n_layers = 1
# model = create_custom_model(n_features, n_classes,
#                             4, n_layers)
# model.summary()

# start_time = time.time()
# print('Model name:', model.name)
# history_callback = model.fit(X_train, Y_train,
#                              batch_size=5,
#                              epochs=400,
#                              verbose=0,
#                              validation_data=(X_test, Y_test)
#                              )
# score = model.evaluate(X_test, Y_test)
# print('Test loss:', score[0])
# print('Test accuracy:', score[1])
# print("--- %s seconds ---" % (time.time() - start_time))


def get_shape(model):
    weights_layer = model.get_weights()
    shapes = []
    for weights in weights_layer:
        shapes.append(weights.shape)
    return shapes

def set_shape(weights,shapes):
    new_weights = []
    index=0
    for shape in shapes:
        if(len(shape)>1):
            n_nodes = np.prod(shape)+index
        else:
            n_nodes=shape[0]+index
        tmp = np.array(weights[index:n_nodes]).reshape(shape)
        new_weights.append(tmp)
        index=n_nodes
    return new_weights

def evaluate_nn(W, shape,X_train=X_train, Y_train=Y_train):
    results = []
    for weights in W:
        model.set_weights(set_shape(weights,shape))
        score = model.evaluate(X_train, Y_train, verbose=0)
        results.append(1-score[1])
    return results

shape = get_shape(model)
x_max = 1.0 * np.ones(83)
x_min = -1.0 * x_max
bounds = (x_min, x_max)
options = {'c1': 0.4, 'c2': 0.8, 'w': 0.4}
optimizer = GlobalBestPSO(n_particles=25, dimensions=83,
                          options=options, bounds=bounds)