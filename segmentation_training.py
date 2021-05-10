import os
import cv2
import json
import time
import keras
import numpy as np
# from Losses import *
from keras.models import *
from keras.callbacks import *
from keras.models import Model
import matplotlib.pyplot as plt
from keras.optimizers import Adam, SGD
from keras.layers.merge import concatenate
from keras.layers import Conv2D, MaxPool2D, Dropout, UpSampling2D, Input, Lambda, Conv2DTranspose, LeakyReLU
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

TRAIN_DIR = './training_data/train/'
VAL_DIR = './training_data/val/'

def rv_iou(y_true, y_pred):

    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)

    intersection = y_true * y_pred

    not_true = 1 - y_true
    union = y_true + (not_true * y_pred)

    #return intersection.sum() / union.sum()
    return 1 - K.sum(intersection) / K.sum(union)

class NumpyDataGenerator(keras.utils.Sequence):
  'Generates data for Keras'
  def __init__(self, images_link, batch_size=32, dim=(512, 512), shuffle=False):
    self.dim = dim
    self.batch_size = batch_size
    self.images_link = images_link
    self.shuffle = shuffle

    self.images = [f for f in os.listdir(self.images_link) if '_mask' not in f]
    print('Data generator initialized on {} samples'.format(len(self.images)))
    self.on_epoch_end()

  def __len__(self):
    'Denotes the number of batches per epoch'
    return int(np.floor(len(self.images) / self.batch_size))

  def __getitem__(self, index):
    'Generate one batch of data'
    #print('Index: {}'.format(index))
    # Generate indexes of the batch
    indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
    list_images_temp = [self.images[k] for k in indexes]

    # Generate data
    # X, y = self.__data_generation(list_images_temp)
    X, y = self.__data_generation(list_images_temp)

    return X, y

  def on_epoch_end(self):
    'Updates indexes after each epoch'
    self.indexes = np.arange(len(self.images))
    if self.shuffle == True:
      np.random.shuffle(self.indexes)

  def __data_generation(self, list_images_temp):
    '''
    Generate data with the specified batch size
    '''
    # Initialization
    X = np.empty((self.batch_size, *self.dim, 1), dtype=np.float32)
    y = np.empty((self.batch_size, *self.dim, 1), dtype=np.uint8)

    for i, image_name in enumerate(list_images_temp):
        image = cv2.imread(self.images_link + image_name)
        mask = cv2.imread(self.images_link + image_name.replace('.png', '_mask.png'))

        X[i, : ,:, 0] = cv2.resize(image[:, :, 0], self.dim) / 255
        y[i, :, :, 0] = cv2.resize(mask[:, :, 0], self.dim) / 255

    return X, y

def unet(input_size = (256,256,1)):
    inputs = Input(input_size)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    pool1 = MaxPool2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    pool2 = MaxPool2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    pool3 = MaxPool2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPool2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
    merge6 = concatenate([drop4,up6], axis = 3)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

    up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
    merge7 = concatenate([conv3,up7], axis = 3)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

    up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
    merge8 = concatenate([conv2,up8], axis = 3)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

    up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
    merge9 = concatenate([conv1,up9], axis = 3)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv9)

    model = Model(input = inputs, output = conv10)

    #model.summary()

    return model

def main():
  train_gen = NumpyDataGenerator(TRAIN_DIR, 2, dim=(256, 256), shuffle=True)
  val_gen = NumpyDataGenerator(VAL_DIR, 2, dim=(256, 256))

  train_model = unet((256, 256, 1))
  train_model.summary()

  train_model.compile(optimizer = Adam(lr = 1e-4), loss = rv_iou, metrics = ['accuracy'])
  reducelr = ReduceLROnPlateau('val_loss', 0.5, 5, verbose = 1)
  checkpoint = ModelCheckpoint('./checkpoints/unet_brain_tumor_2.h5', 'val_loss', save_best_only = True, verbose = 1)

  train_model.load_weights('./checkpoints/unet_brain_tumor_1.h5')

  history = train_model.fit_generator(train_gen, train_gen.__len__(), 100, callbacks = [reducelr, checkpoint], validation_data=val_gen)
  plt.plot(history.history['val_loss'])
  plt.show()

if __name__ == '__main__':
  main()