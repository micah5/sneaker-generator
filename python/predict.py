import matplotlib
matplotlib.use('Agg')

import time
import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

from keras.models import Sequential
from keras.layers import Conv2D, Conv2DTranspose, Reshape
from keras.layers import Flatten, BatchNormalization, Dense, Activation
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator

import tensorflowjs as tfjs

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def construct_generator():

    generator = Sequential()

    generator.add(Dense(units=16 * 16 * 512,
                        kernel_initializer='glorot_uniform',
                        input_shape=(1, 1, 100)))
    generator.add(Reshape(target_shape=(16, 16, 512)))
    generator.add(BatchNormalization(momentum=0.5))
    generator.add(Activation('relu'))

    generator.add(Conv2DTranspose(filters=256, kernel_size=(5, 5),
                                  strides=(2, 2), padding='same',
                                  data_format='channels_last',
                                  kernel_initializer='glorot_uniform'))
    generator.add(BatchNormalization(momentum=0.5))
    generator.add(Activation('relu'))

    generator.add(Conv2DTranspose(filters=128, kernel_size=(5, 5),
                                  strides=(2, 2), padding='same',
                                  data_format='channels_last',
                                  kernel_initializer='glorot_uniform'))
    generator.add(BatchNormalization(momentum=0.5))
    generator.add(Activation('relu'))

    generator.add(Conv2DTranspose(filters=64, kernel_size=(5, 5),
                                  strides=(2, 2), padding='same',
                                  data_format='channels_last',
                                  kernel_initializer='glorot_uniform'))
    generator.add(BatchNormalization(momentum=0.5))
    generator.add(Activation('relu'))

    generator.add(Conv2DTranspose(filters=3, kernel_size=(5, 5),
                                  strides=(2, 2), padding='same',
                                  data_format='channels_last',
                                  kernel_initializer='glorot_uniform'))
    generator.add(Activation('tanh'))

    optimizer = Adam(lr=0.00015, beta_1=0.5)
    generator.compile(loss='binary_crossentropy',
                      optimizer=optimizer,
                      metrics=None)

    return generator

def predict():
    generator = construct_generator()

    if os.path.exists("./output/generator_weights.h5"):
        print('loaded generator model weights')
        generator.load_weights('./output/generator_weights.h5')

    # Saving model for tensorflow.js
    tfjs.converters.save_keras_model(generator, 'generator')

    batch_size = 64

    # Generate noise
    noise = np.random.uniform(size=[batch_size, 1, 1, 100])
    print(noise.shape)

    # Generate images
    generated_images = generator.predict(noise)

    # Save images
    for i in range(batch_size):
        image = generated_images[i, :, :, :]
        image += 1
        image *= 127.5
        matplotlib.image.imsave('./output/shoe%d.png' % i, image.astype(np.uint8))

predict()
