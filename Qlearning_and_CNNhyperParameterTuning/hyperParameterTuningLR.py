# ------------------------------------------- WSU - CPT_S 570 - HW4 -----------------------------------------
# -------------------------------------- Hyper parameter tuning for CNN model -----------------------------------
# ----------------------------------------- Student: Francisco Goncalves ------------------------------------------
# Tunes Learning Rate

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from tensorflow.keras import datasets, layers
from convNN import num_classes, test_ds, train_ds

model1 = tf.keras.Sequential([
    layers.ZeroPadding2D(padding=2, input_shape=(28, 28, 1)),
    layers.Conv2D(8, kernel_size=5, strides=2, activation='relu'),
    layers.ZeroPadding2D(padding=1, data_format=None),
    layers.Conv2D(16, kernel_size=3, strides=2, activation='relu'),
    layers.ZeroPadding2D(padding=1, data_format=None),
    layers.Conv2D(32, kernel_size=3, strides=2, activation='relu'),
    layers.ZeroPadding2D(padding=1, data_format=None),
    layers.Conv2D(32, kernel_size=3, strides=2, activation='relu'),
    layers.AveragePooling2D(),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(num_classes),
    layers.Softmax()
])
model2 = tf.keras.Sequential([
    layers.ZeroPadding2D(padding=2, input_shape=(28, 28, 1)),
    layers.Conv2D(8, kernel_size=5, strides=2, activation='relu'),
    layers.ZeroPadding2D(padding=1),
    layers.Conv2D(16, kernel_size=3, strides=2, activation='relu'),
    layers.ZeroPadding2D(padding=1),
    layers.Conv2D(32, kernel_size=3, strides=2, activation='relu'),
    layers.ZeroPadding2D(padding=1),
    layers.Conv2D(32, kernel_size=3, strides=2, activation='relu'),
    layers.AveragePooling2D(),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(num_classes),
    layers.Softmax()
])

opt = tf.keras.optimizers.Adam(learning_rate=1)
model1.compile(optimizer=opt,
               loss=tf.keras.losses.SparseCategoricalCrossentropy(),
               metrics=['accuracy'])

history1 = model1.fit(train_ds, epochs=20,
                      validation_data=test_ds)

opt = tf.keras.optimizers.Adam(learning_rate=0.001)
model2.compile(optimizer=opt,
               loss=tf.keras.losses.SparseCategoricalCrossentropy(),
               metrics=['accuracy'])

history2 = model2.fit(train_ds, epochs=20,
                      validation_data=test_ds)

plt.plot(history1.history['loss'], "k", label='lr=1')
plt.plot(history2.history['loss'], "b", label='lr=1e-3')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='lower right')
plt.figure()
plt.plot(history1.history['loss'], "k", label='lr=1')
plt.plot(history2.history['loss'], "b", label='lr=1e-3')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.ylim([0, 3])
plt.legend(loc='lower right')
plt.figure()
plt.plot(history1.history['val_accuracy'], ":xk", label='Tst accuracy1 lr=1')
plt.plot(history2.history['val_accuracy'], "-dk", label='Tst accuracy lr=1e-3')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')