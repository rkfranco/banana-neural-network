import os

import cv2 as cv
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tensorflow import keras

nomes_classes = ["rotten", "healthy"]


def load_data_path(path, class_type, x, y):
    for img_path in os.listdir(path):
        img = cv.imread(f'{path}/{img_path}', cv.IMREAD_GRAYSCALE)
        x.append(cv.resize(img, (224, 224)) / 255.0)
        y.append(class_type)


def load_dataset():
    x_train = []
    y_train = []
    x_test = []
    y_test = []

    load_data_path("data/train/Banana__Healthy", 1, x_train, y_train)
    load_data_path("data/train/Banana__Rotten", 0, x_train, y_train)
    load_data_path("data/test/Banana__Healthy", 1, x_test, y_test)
    load_data_path("data/test/Banana__Rotten", 0, x_test, y_test)

    return np.array(x_train), np.array(y_train), np.array(x_test), np.array(y_test)


if __name__ == '__main__':
    x_train, y_train, x_test, y_test = load_dataset()

    nn = keras.models.Sequential([
        keras.layers.InputLayer(input_shape=(224, 224, 1)),
        keras.layers.Conv2D(64, 5, activation='relu', padding='same', kernel_initializer='glorot_uniform'),
        keras.layers.MaxPool2D(2),
        keras.layers.Conv2D(128, 3, activation='relu', padding='same'),
        keras.layers.Conv2D(128, 3, activation='relu', padding='same'),
        keras.layers.MaxPool2D(2),
        keras.layers.Conv2D(258, 3, activation='relu', padding='same'),
        keras.layers.Conv2D(258, 3, activation='relu', padding='same'),
        keras.layers.MaxPool2D(2),
        keras.layers.Flatten(),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dropout(rate=0.5),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dropout(rate=0.5),
        keras.layers.Dense(2, activation="softmax")])

    X_train_new = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
    X_test_new = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)

    nn.compile(loss="sparse_categorical_crossentropy",
               optimizer="sgd",
               metrics=["accuracy"])
    history_nn = nn.fit(X_train_new, y_train, epochs=20, validation_data=(X_test_new, y_test))

    pd.DataFrame(history_nn.history).plot(figsize=(12, 8))
    plt.grid(True)
    plt.gca().set_ylim(0, 1)
    plt.show()

    result = nn.predict(x_test[:10])

    for i in range(len(result)):
        print(f'Previsao: {np.argmax(result[i], axis=-1)} | Verdadeiro: {y_test[i]}')

    result = nn.predict(x_test[350:])

    for i in range(len(result)):
        print(f'Previsao: {np.argmax(result[i], axis=-1)} | Verdadeiro: {y_test[350 + i]}')

    nn.evaluate(X_test_new, y_test, verbose=1)
