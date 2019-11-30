from __future__ import absolute_import, division, print_function, unicode_literals
import os
import keras
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, Convolution2D, GlobalAveragePooling2D
from tensorflow.keras.datasets.cifar10 import load_data
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot 
import numpy as np
import sys


def summarize_diagnostics(history):
	# plot loss
	pyplot.subplot(211)
	pyplot.title('Cross Entropy Loss')
	pyplot.plot(history.history['loss'], color='blue', label='train')
	pyplot.plot(history.history['val_loss'], color='orange', label='test')
	# plot accuracy
	pyplot.subplot(212)
	pyplot.title('Classification Accuracy')
	pyplot.plot(history.history['acc'], color='blue', label='train')
	pyplot.plot(history.history['val_acc'], color='orange', label='test')
	# save plot to file
	filename = sys.argv[0].split('/')[-1]
	pyplot.savefig(filename + '_plot.png')
	pyplot.close()

	
(train_images, train_labels), (test_images, test_labels) = load_data()

train_images = train_images.reshape((50000, 32, 32, 3))

test_images = test_images.reshape((10000, 32, 32, 3))

# 픽셀 값을 0~1 사이로 정규화합니다.
train_images, test_images = train_images / 255.0, test_images / 255.0

if os.path.isfile("cifar-10cnn_model.h5"):
    model=tf.keras.models.load_model('cifar-10cnn_model.h5')



else:
    model = Sequential([
        Conv2D(96, 3, padding='same', activation='relu', input_shape=(32, 32 ,3)),
        Conv2D(96, 3, padding='same', activation='relu'),
        Conv2D(96, 3, padding='same'),
        MaxPooling2D(),
        Dropout(0.5),   
        Conv2D(192, 3, padding='same', activation='relu'),
        Conv2D(192, 3, padding='same', activation='relu'),
        Conv2D(192, 3, padding='same'),
        MaxPooling2D(),
        Dropout(0.5),
        Conv2D(192, 3, padding='same', activation='relu'),
        Conv2D(192, 1, padding='valid', activation='relu'),
        Conv2D(10, 1, padding='valid'),
        GlobalAveragePooling2D(),
        Dense(10, activation='softmax')
    ])

datagen = ImageDataGenerator(rotation_range=0,width_shift_range=0.1, height_shift_range = 0.1, horizontal_flip =True)
datagen.fit(train_images)
it_train = datagen.flow(train_images,train_labels,32)
opt=tf.keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

model.compile(optimizer=opt,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


hist = model.fit_generator(it_train, steps_per_epoch=train_images.shape[0]//32,epochs=350,verbose=1, validation_data=(test_images,test_labels),)

test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
print(test_acc)
model.save('cifar-10cnn_model.h5')
summarize_diagnostics(hist)

