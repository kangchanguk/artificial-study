from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.datasets.cifar10 import load_data

import numpy as np
#from tensorflow.keras import datasets, layers, models

#(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()
(train_images, train_labels), (test_images, test_labels) = load_data()

#이미지 포맷이 흑백에 사이즈 28*28입니다.
train_images = train_images.reshape((50000, 32, 32, 3))

test_images = test_images.reshape((10000, 32, 32, 3))

# 픽셀 값을 0~1 사이로 정규화합니다.
train_images, test_images = train_images / 255.0, test_images / 255.0

model = Sequential([
    Conv2D(32, 3, padding='same', activation='relu', input_shape=(32, 32 ,3)),
    Conv2D(32, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Dropout(0.2),
    Conv2D(64, 3, padding='same', activation='relu'),
    Conv2D(64, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Dropout(0.2),
    Conv2D(128, 3, padding='same', activation='relu'),
    Conv2D(128, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Dropout(0.2),
    Conv2D(256, 3, padding='same', activation='relu'),
    Conv2D(256, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

model.compile(optimizer="adam",
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=50)
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
print(test_acc)
