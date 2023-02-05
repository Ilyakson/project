import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.applications import MobileNet
from keras.models import Sequential
from keras.layers import (
    Conv2D, MaxPool2D, Dropout,
    Dense, Flatten, MaxPooling2D
)
from keras.callbacks import EarlyStopping
from keras.layers import Rescaling


path = "/Users/Илья Друзь/Projects/ml/chess"
train_data = tf.keras.preprocessing.image_dataset_from_directory(
    path, validation_split=0.2,
    image_size=(224, 224),
    batch_size=32,
    subset="training",
    seed=123
)

test_data = tf.keras.preprocessing.image_dataset_from_directory(
    path, validation_split=0.2,
    image_size=(224, 224),
    batch_size=32,
    subset="validation",
    seed=123
)
class_names = ['Queen', 'Rook', 'bishop', 'knight', 'pawn']
inputs = tf.keras.Input(shape=(224, 224, 3))
preprocess = tf.keras.applications.mobilenet.preprocess_input(inputs)
upscale = tf.keras.layers.Lambda(
    lambda x: tf.image.resize_with_pad(
        x,
        224,
        224,
        method=tf.image.ResizeMethod.BILINEAR
    )
)(inputs)
mobilenet = MobileNet(
    include_top='True',
    weights='imagenet',
    input_tensor=upscale,
    input_shape=(224, 224, 3)
)
data_augmentation = tf.keras.Sequential(
  [
    tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal"),
    tf.keras.layers.experimental.preprocessing.RandomRotation(0.1),
    tf.keras.layers.experimental.preprocessing.RandomZoom(0.1),
  ]
)
model = Sequential()
model.add(Rescaling(1./255, input_shape=(224, 224, 3)))
model.add(data_augmentation)
model.add(mobilenet)
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(len(class_names), activation='softmax'))
model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    optimizer='adam',
    metrics=['accuracy']
)

model.summary()
mycallbacks = [EarlyStopping(
    monitor='val_loss',
    patience=20,
    restore_best_weights=True
)]
hist = model.fit(
    train_data,
    validation_data=test_data,
    epochs=100,
    callbacks=mycallbacks
)
plt.figure()
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.legend(
    ['loss', 'val_loss'],
    loc='upper right'
)
plt.show()
