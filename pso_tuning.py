# %%
import json
from tqdm import tqdm
from datetime import date
import matplotlib.pyplot as plt
import numpy as np
from PSO.pso_bp import PSO
from keras import backend as K
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Dropout, Flatten
from keras.models import Sequential
from keras.datasets import mnist
from tensorflow import keras
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

tf.random.set_seed(777)  # for reproducibility


print(tf.__version__)
print(tf.config.list_physical_devices())


def get_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train, x_test = x_train / 255.0, x_test / 255.0
    x_train = x_train.reshape((60000, 28, 28, 1))
    x_test = x_test.reshape((10000, 28, 28, 1))

    print(f"x_train : {x_train[0].shape} | y_train : {y_train[0].shape}")
    print(f"x_test : {x_test[0].shape} | y_test : {y_test[0].shape}")
    return x_train, y_train, x_test, y_test


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

    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])

    # model.summary()

    return model


# %%
'''
optimizer parameter
'''
lr = 0.1
momentun = 0.8
decay = 1e-04
nestrov = True

'''
pso parameter
'''
n_particles = 100
maxiter = 500
# epochs = 1
w = 0.8
c0 = 0.6
c1 = 1.6

def auto_tuning():
    x_train, y_train, x_test, y_test = get_data()
    model = make_model()

    loss = keras.losses.MeanSquaredError()
    optimizer = keras.optimizers.SGD(lr=lr, momentum=momentun, decay=decay, nesterov=nestrov)


    pso_m = PSO(model=model, loss_method=loss, n_particles=n_particles)
    # c0 : 지역 최적값 중요도
    # c1 : 전역 최적값 중요도
    # w : 관성 (현재 속도를 유지하는 정도)
    best_weights, score = pso_m.optimize(x_train, y_train, x_test, y_test, maxiter=maxiter, c0=c0, c1=c1, w=w)
    model.set_weights(best_weights)

    score_ = model.evaluate(x_test, y_test, verbose=2)
    print(f" Test loss: {score_}")
    score = round(score_[0]*100, 2)

    day = date.today().strftime("%Y-%m-%d")

    model.save(f'./model/{day}_{score}_mnist.h5')
    json_save = {
        "name" : f"{day}_{score}_mnist.h5",
        "score" : score_,
        "maxiter" : maxiter,
        "c0" : c0,
        "c1" : c1,
        "w" : w    
    }
    with open(f'./model/{day}_{score}_bp_mnist.json', 'a') as f:
        json.dump(json_save, f)
        f.write(',\n')
                    
    return model
auto_tuning()


# %%
# print(f"정답 > {y_test}")
def get_score(model):
    x_train, y_train, x_test, y_test = get_data()

    predicted_result = model.predict(x_test)
    predicted_labels = np.argmax(predicted_result, axis=1)
    not_correct = []
    for i in tqdm(range(len(y_test)), desc="진행도"):
        if predicted_labels[i] != y_test[i]:
            not_correct.append(i)
            # print(f"추론 > {predicted_labels[i]} | 정답 > {y_test[i]}")

    print(f"틀린 갯수 > {len(not_correct)}/{len(y_test)}")

    for i in range(3):
        plt.imshow(x_test[not_correct[i]].reshape(28, 28), cmap='Greys')
    plt.show()

# %%


def default_mnist(epochs=5):
    x_train, y_train, x_test, y_test = get_data()
    model = make_model()

    hist = model.fit(x_train, y_train, epochs=epochs, batch_size=32, verbose=1)
    print(hist.history['loss'][-1])
    print(hist.history['accuracy'][-1])

    predicted_result = model.predict(x_test)
    predicted_labels = np.argmax(predicted_result, axis=1)
    not_correct = []
    for i in tqdm(range(len(y_test)), desc="진행도"):
        if predicted_labels[i] != y_test[i]:
            not_correct.append(i)
            # print(f"추론 > {predicted_labels[i]} | 정답 > {y_test[i]}")

    print(f"틀린 갯수 > {len(not_correct)}/{len(y_test)}")


# %%
