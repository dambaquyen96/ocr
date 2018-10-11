#!/usr/bin/env python
# coding: utf-8

import keras
import numpy as np
import os
import pickle
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=False,
        vertical_flip=False,
        fill_mode='nearest')

test_datagen = ImageDataGenerator(rescale=1./255)


BATCH_SIZE = 128
N_CLASSES = 3095
LR = 0.001
N_EPOCHS = 50
N_UNITS = 128
IMG_SIZE = 100


train_generator = train_datagen.flow_from_directory(
        'dataset/print_3095/training_dataset',
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='sparse')


validation_generator = test_datagen.flow_from_directory(
        'dataset/print_3095/testing_dataset',
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='sparse')

from keras.applications.mobilenet import MobileNet
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D
from keras.models import Sequential, Model 

model = MobileNet(input_shape=(IMG_SIZE, IMG_SIZE, 3), include_top=True, classes=N_CLASSES, weights=None)
model.summary()

# Training
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adadelta

model_file = "models/weights-improvement-{epoch:02d}-{val_loss:.2f}.hdf5"

checkpoint = ModelCheckpoint(model_file, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
tbCallBack = keras.callbacks.TensorBoard(log_dir='logs/tensorboard', write_graph=True, write_images=True)

callbacks_list = [checkpoint, tbCallBack]
# model.load_weights("models/weights-improvement-16-0.05.hdf5")
model.compile(loss=keras.losses.sparse_categorical_crossentropy,
              optimizer=Adadelta(),
              metrics=['accuracy'])

model.fit_generator(
        train_generator,
        steps_per_epoch=479725 // BATCH_SIZE,
        initial_epoch=0,
        epochs=50,
        validation_data=validation_generator,
        validation_steps=64995 // BATCH_SIZE,
        callbacks=callbacks_list,
        max_queue_size=60,
        workers=30,
        use_multiprocessing=True,
        )





