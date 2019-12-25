import tensorflow as tf
from tensorflow.keras.layers import Conv3D
from tensorflow import nn
from tensorflow.python.ops import nn_ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras.engine.base_layer import InputSpec
from tensorflow.python.keras.utils import conv_utils

from tensorflow.keras import Model

class SimpleCNN(Model):
  def __init__(self, cat_cols, num_cols):
    super(SimpleCNN, self).__init__()
    self.cat_cols = cat_cols
    self.num_cols = num_cols
    self.ac = tf.keras.layers.ReLU()
    self.maxpool = tf.keras.layers.MaxPool3D(pool_size=(2, 2, 2), data_format='channels_last')
    self.conv1 = tf.keras.layers.Conv3D(
        filters = 32,
        kernel_size = 3,
        padding='valid',
        data_format='channels_last'
    )
    self.bn1 = tf.keras.layers.BatchNormalization()
    self.conv2 = tf.keras.layers.Conv3D(
        filters = 64,
        kernel_size = 3,
        padding='valid',
        data_format='channels_last'
    )
    self.bn2 = tf.keras.layers.BatchNormalization()
    self.conv3 = tf.keras.layers.Conv3D(
        filters = 128,
        kernel_size = 3,
        padding='valid',
        data_format='channels_last'
    )
    self.bn3 = tf.keras.layers.BatchNormalization()
    self.conv4 = tf.keras.layers.Conv3D(
        filters = 256,
        kernel_size = 3,
        padding='valid',
        data_format='channels_last'
    )
    self.bn4 = tf.keras.layers.BatchNormalization()
    self.fc = {}
    for k in list(self.cat_cols.keys()):
      self.fc[k] = [
          tf.keras.layers.Dense(256, activation='relu'),
          tf.keras.layers.BatchNormalization(),
          tf.keras.layers.Dense(self.cat_cols[k], activation='softmax')
      ]
    for i in range(len(self.num_cols)):
      self.fc[self.num_cols[i]] = [
          tf.keras.layers.Dense(256, activation='relu'),
          tf.keras.layers.BatchNormalization(),
          tf.keras.layers.Dense(1)
      ]

  def call(self, x):
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.ac(x)
    x = self.maxpool(x)
    x = self.conv2(x)
    x = self.bn2(x)
    x = self.ac(x)
    x = self.maxpool(x)
    x = self.conv3(x)
    x = self.bn3(x)
    x = self.ac(x)
    x = self.maxpool(x)
    x = self.conv4(x)
    x = self.bn4(x)
    x = self.ac(x)
    x = tf.keras.layers.GlobalAveragePooling3D()(x)
    out = {}
    for k in list(self.fc.keys()):
      out[k] = self.fc[k][2](self.fc[k][1](self.fc[k][0](x)))
    return out