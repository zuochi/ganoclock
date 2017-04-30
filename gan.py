#!/usr/bin/env python
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

from scipy.misc import imsave
import numpy as np
import random
from keras.datasets import mnist
from keras.models import Model
from keras.layers import Input, Activation, Dense, Reshape, Flatten
from keras.layers.convolutional import Conv2D, UpSampling2D, MaxPooling2D
from keras.layers.advanced_activations import ELU

GEN_SIZE = 16
BATCH_SIZE = 64

def encoder_model():
  layers = []
  layers.append(Conv2D(4, (3, 3), padding='same', input_shape=(28,28,1)))
  for f in [16,32,64]:
    layers.append(Conv2D(f, (3, 3), strides=(2,2), padding='same'))
    layers.append(ELU())
  layers.append(Flatten())
  layers.append(Dense(256))
  layers.append(ELU())
  layers.append(Dense(GEN_SIZE))
  layers.append(Activation('tanh'))
  return layers

def generator_model():
  layers = []
  layers.append(Dense(64, input_dim=GEN_SIZE))
  layers.append(ELU())
  layers.append(Dense(7*7*4))
  layers.append(ELU())
  layers.append(Reshape((7,7,4)))
  for f in [32,16]:
    layers.append(Conv2D(f, (3, 3), padding='same'))
    layers.append(ELU())
    layers.append(UpSampling2D(size=(2,2)))
  layers.append(Conv2D(1, (3, 3), padding='same'))
  layers.append(Activation('sigmoid'))
  return layers

def discriminator_model():
  layers = []
  layers.append(Conv2D(6, (3, 3), input_shape=(28,28,1)))
  for f in [16,24,32]:
    layers.append(Conv2D(f, (3, 3), strides=(2,2)))
    layers.append(ELU())
  layers.append(Flatten())
  layers.append(Dense(64))
  layers.append(ELU())
  layers.append(Dense(1))
  layers.append(Activation('sigmoid'))
  return layers

def main():
  (x_train, y_train), (x_test, y_test) = mnist.load_data()

  # **** create models ****
  l_discriminator = discriminator_model()
  l_encoder = encoder_model()
  l_generator = generator_model()

  # *** create encoder ***
  img = Input(shape=(28, 28, 1), name="input_img1")
  x = img
  for l in l_encoder:
    x = l(x)
  encoder = Model(inputs=[img], outputs=[x])
  encoder.summary()

  # *** create generator ***
  feat = Input(shape=(GEN_SIZE,))
  x = feat
  for l in l_generator:
    x = l(x)
  generator = Model(inputs=[feat], outputs=[x])
  generator.summary()

  # *** create discriminator ***
  img = Input(shape=(28, 28, 1), name="input_img1")
  x = img
  for l in l_discriminator:
    x = l(x)
  discriminator = Model(inputs=[img], outputs=[x])
  discriminator.compile(loss='binary_crossentropy', optimizer='adam')
  discriminator.summary()

  # *** create generator-discriminator ***
  feat = Input(shape=(GEN_SIZE,))
  x = feat
  for l in l_generator:
    x = l(x)
  for l in l_discriminator:
    l.trainable = False
    x = l(x)
  generator_discriminator = Model(inputs=[feat], outputs=[x])
  generator_discriminator.compile(loss='binary_crossentropy', optimizer='adam')
  generator_discriminator.summary()

  # *** create encoder-generator ***
  img = Input(shape=(28, 28, 1), name="input_img1")
  x = img
  for l in l_encoder:
    x = l(x)
  for l in l_generator:
    x = l(x)
  encoder_generator = Model(inputs=[img], outputs=[x])
  encoder_generator.compile(loss='mse', optimizer='adam')
  encoder_generator.summary()

  def write_images(lst):
    gg = [np.clip(x[0:8] * 255, 0, 255).astype(np.uint8) for x in lst]
    print gg[0].shape
    rgbs = [np.concatenate(x, axis=0) for x in gg]
    bigimg = np.concatenate(rgbs, axis=1)
    bigimg = np.repeat(bigimg, 3, 2)
    print bigimg.shape
    imsave('/tmp/test.png', bigimg)

  def get_real_batch():
    idxs = [random.randint(0, x_train.shape[0]-1) for x in range(BATCH_SIZE)]
    return x_train[idxs].reshape((64, 28, 28, 1)).astype('float32')/255

  d_loss, g_loss, eg_loss = 0.0, 0.0, 0.0
  idx = 0
  while 1:
    # get real images
    real_images = get_real_batch()

    # generate fake images
    noise = np.random.normal(size=(BATCH_SIZE, GEN_SIZE))
    generated_images = generator.predict(noise, verbose=0)

    if idx%0x100 == 0:
      # generate ae images
      ae_generated_images = encoder_generator.predict(real_images)

      write_images([generated_images, ae_generated_images, real_images])

    # it's train o'clock
    eg_loss = encoder_generator.train_on_batch(real_images, real_images)

    if eg_loss < 0.05:
      d_loss = discriminator.train_on_batch(
                 np.concatenate([real_images, generated_images], axis=0),
                 [1] * BATCH_SIZE + [0] * BATCH_SIZE)

      g_loss = generator_discriminator.train_on_batch(noise, [1] * BATCH_SIZE)


    print "%10d   d: %.4f   g: %.4f   eg: %.4f" % (idx, d_loss, g_loss, eg_loss)

    idx += 1

if __name__ == "__main__":
  main()

