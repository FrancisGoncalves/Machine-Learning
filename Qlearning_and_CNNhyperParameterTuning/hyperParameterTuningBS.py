# ------------------------------------------- WSU - CPT_S 570 - HW4 -----------------------------------------
# -------------------------------------- Hyper parameter tuning for CNN model -----------------------------------
# ----------------------------------------- Student: Francisco Goncalves ------------------------------------------
# Tunes Batch Size
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from tensorflow.keras import datasets, layers
from convNN import num_classes, test_ds, X_train, y_train

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
train_ds1 = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(1000)
train_ds2 = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(100)

model1.compile(optimizer='adam',
               loss=tf.keras.losses.SparseCategoricalCrossentropy(),
               metrics=['accuracy'])
model2.compile(optimizer='adam',
               loss=tf.keras.losses.SparseCategoricalCrossentropy(),
               metrics=['accuracy'])

history1 = model1.fit(train_ds1, epochs=5,
                      validation_data=test_ds)

history2 = model2.fit(train_ds2, epochs=5,
                      validation_data=test_ds)

plt.plot(history1.history['loss'], "k", label='Batch 1e3')
plt.plot(history2.history['loss'], "b", label='Batch 1e2')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='lower right')
plt.figure()
plt.plot(history1.history['val_accuracy'], ":xk", label='Batch 1e3')
plt.plot(history2.history['val_accuracy'], "-dk", label='Batch 1e2')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')