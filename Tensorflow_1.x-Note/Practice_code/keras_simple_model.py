from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
from tensorflow.keras import layers
import numpy as np

model = tf.keras.Sequential([

layers.Dense(64, activation='relu', input_shape=(32,)),
layers.Dense(64, activation='relu'),
layers.Dense(10, activation='softmax')])

model.compile(optimizer=tf.train.AdamOptimizer(0.001), loss='categorical_crossentropy',metrics=['accuracy'])

def random_one_hot_labels(shape):
  n, n_class = shape
  classes = np.random.randint(0, n_class, n)
  labels = np.zeros((n, n_class))
  labels[np.arange(n), classes] = 1
  return labels

data = np.random.random((1000, 32))
labels = random_one_hot_labels((1000, 10))

val_data = np.random.random((100, 32))
val_labels = random_one_hot_labels((100, 10))

model.fit(data, labels, epochs=100, batch_size=32, validation_data=(val_data, val_labels))

model.evaluate(data, labels, batch_size=32)

result = model.predict(data, batch_size=32)

