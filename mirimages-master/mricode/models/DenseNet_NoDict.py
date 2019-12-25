import tensorflow as tf
from tensorflow.keras.layers import Conv3D
from tensorflow import nn
from tensorflow.python.ops import nn_ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras.engine.base_layer import InputSpec
from tensorflow.python.keras.utils import conv_utils

from tensorflow.keras import Model

class MyDenseNet(Model):
  def __init__(self, cat_cols, num_cols):
    super(MyDenseNet, self).__init__()
    self.cat_cols = cat_cols
    self.num_cols = num_cols
    self.fc = {}
    for k in list(self.cat_cols.keys()):
      self.fc[k] = [
          tf.keras.layers.Dense(256, activation='relu'),
          tf.keras.layers.BatchNormalization(),
          tf.keras.layers.Dense(self.cat_cols[k], activation='softmax')]
    for i in range(len(self.num_cols)):
      self.fc[self.num_cols[i]] = [
          tf.keras.layers.Dense(256, activation='relu'),
          tf.keras.layers.BatchNormalization(),
          tf.keras.layers.Dense(1)]

    self.ac = tf.keras.layers.ReLU()
    self.conv1 = tf.keras.layers.Conv3D(filters = 32,
                                        kernel_size = 7,
                                        padding='same',
                                        data_format='channels_last'
                                      )

    self.bottleneck1 = []
    start_filter_num1 = 16
    for b in range(4):
      filter_num1 = int(start_filter_num1*0.5)
      filter_num2 = int(filter_num1*0.5)
      start_filter_num1 += filter_num2
      self.bottleneck1.append(tf.keras.Sequential([
                                                tf.keras.layers.BatchNormalization(),
                                                 tf.keras.layers.ReLU(),
                                                 tf.keras.layers.Conv3D(filters = filter_num1,
                                                  kernel_size = 1,
                                                  padding='same',
                                                  data_format='channels_last'),
                                                 tf.keras.layers.BatchNormalization(),
                                                 tf.keras.layers.ReLU(),
                                                 tf.keras.layers.Conv3D(filters = filter_num2,
                                                  kernel_size = 3,
                                                  padding='same',
                                                  data_format='channels_last')]))
      
    start_filter_num2 = int(start_filter_num1 * 0.5)
    self.bottleneck2 = []
    for b in range(4):
      filter_num1 = int(start_filter_num2*0.5)
      filter_num2 = int(filter_num1*0.5)
      start_filter_num1 += filter_num2
      self.bottleneck2.append(tf.keras.Sequential([
                                                tf.keras.layers.BatchNormalization(),
                                                tf.keras.layers.ReLU(),
                                                tf.keras.layers.Conv3D(filters = filter_num1,
                                                  kernel_size = 1,
                                                  padding='same',
                                                  data_format='channels_last'),
                                                tf.keras.layers.BatchNormalization(),
                                                tf.keras.layers.ReLU(),
                                                tf.keras.layers.Conv3D(filters = filter_num2,
                                                  kernel_size = 3,
                                                  padding='same',
                                                  data_format='channels_last')
                                                  ]))
    self.bottleneck3 = []
    start_filter_num3 = int(start_filter_num2 * 0.5)
    for b in range(4):
      filter_num1 = int(start_filter_num3*0.5)
      filter_num2 = int(filter_num1*0.5)
      start_filter_num1 += filter_num2
      self.bottleneck3.append(tf.keras.Sequential([
                                                tf.keras.layers.BatchNormalization(),
                                                tf.keras.layers.ReLU(),
                                                tf.keras.layers.Conv3D(filters = 24,
                                                  kernel_size = 1,
                                                  padding='same',
                                                  data_format='channels_last'),
                                                tf.keras.layers.BatchNormalization(),
                                                tf.keras.layers.ReLU(),
                                                tf.keras.layers.Conv3D(filters = 12,
                                                  kernel_size = 3,
                                                  padding='same',
                                                  data_format='channels_last')
                                                  ]))
                                                                                            
  
    self.transition1 = tf.keras.Sequential([
                                              tf.keras.layers.BatchNormalization(),
                                                tf.keras.layers.ReLU(),
                                                tf.keras.layers.Conv3D(filters = int(start_filter_num1*0.5),
                                                kernel_size = 1,
                                                padding='same',
                                                data_format='channels_last'),
                                                tf.keras.layers.AveragePooling3D(data_format='channels_last')
    ])

    self.transition2 = tf.keras.Sequential([
                                              tf.keras.layers.BatchNormalization(),
                                                tf.keras.layers.ReLU(),
                                                tf.keras.layers.Conv3D(filters = start_filter_num2,
                                                kernel_size = 1,
                                                padding='same',
                                                data_format='channels_last'),
                                                tf.keras.layers.AveragePooling3D(data_format='channels_last')
    ])

    self.transition3 = tf.keras.Sequential([
                                              tf.keras.layers.BatchNormalization(),
                                                tf.keras.layers.ReLU(),
                                                tf.keras.layers.Conv3D(filters = start_filter_num3,
                                                kernel_size = 1,
                                                padding='same',
                                                data_format='channels_last'),
                                                tf.keras.layers.AveragePooling3D(data_format='channels_last')
    ])

    self.bn1 = tf.keras.layers.BatchNormalization()
     


  def call(self, x):
    x = self.conv1(x)
    # print("initial shape: ", x.shape)

    ## dense block 1 ##
    num_blocks = 4
    layers_concat = list()
    layers_concat.append(x)
  
    x = self.bottleneck1[0](x)
    # print("first bottleneck shape: ", x.shape)
    layers_concat.append(x)
    
    for d in range(1, num_blocks):
      x = tf.concat(layers_concat, axis=4)
      # print("first concat: ", x.shape)
      x = self.bottleneck1[d](x)
      layers_concat.append(x)
    x = tf.concat(layers_concat, axis=4)
    ## end dense block 1 ##

    x = self.transition1(x)
    # print("first transition shape: ", x.shape)

    ## dense block 2 ##
    num_blocks = 4
    layers_concat = list()
    layers_concat.append(x)
  
    x = self.bottleneck2[0](x)
    layers_concat.append(x)
    
    for d in range(1, num_blocks):
      x = tf.concat(layers_concat, axis=4)
      x = self.bottleneck2[d](x)
      layers_concat.append(x)
    x = tf.concat(layers_concat, axis=4)
    ## end dense block 2 ##

    x = self.transition2(x)

    ## dense block 3 ##
    num_blocks = 4
    layers_concat = list()
    layers_concat.append(x)
  
    x = self.bottleneck3[0](x)
    layers_concat.append(x)
    
    for d in range(1, num_blocks):
      x = tf.concat(layers_concat, axis=4)
      x = self.bottleneck3[d](x)
      layers_concat.append(x)
    x = tf.concat(layers_concat, axis=4)
    ## end dense block 3 ##

    x = self.transition3(x)

    # END PART
    x = self.bn1(x)
    x = self.ac(x)
    x = tf.keras.layers.GlobalAveragePooling3D()(x)
    x = tf.keras.layers.Flatten()(x)
    out = {}
    for k in list(self.fc.keys()):
      out[k] = self.fc[k][2](self.fc[k][1](self.fc[k][0](x)))
    return out