import os

import cv2 as cv
import keras
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

nomes_classes = ["rotten", "healthy"]


def load_data_path(path, classType, x, y):
    for img_path in os.listdir(path):
        img = cv.imread(f'{path}/{img_path}', cv.IMREAD_GRAYSCALE)
        x.append(cv.resize(img, (224, 224)) / 255.0)
        y.append(classType)


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

    """#  Rede Neural Artificial com Keras"""
    nn = keras.models.Sequential()
    nn.add(keras.layers.Flatten(input_shape=[224, 224]))
    nn.add(keras.layers.Dense(300, activation="relu"))
    nn.add(keras.layers.Dropout(rate=0.1))
    nn.add(keras.layers.Dense(100, activation="relu"))
    nn.add(keras.layers.Dropout(rate=0.1))
    nn.add(keras.layers.Dense(2, activation="softmax"))
    print(nn.summary())

    """### Compilando e treinando o modelo"""

    nn.compile(loss="sparse_categorical_crossentropy",
               optimizer="sgd",
               metrics=["accuracy"])
    history_nn = nn.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))

    """### Visualizar a performance"""

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
