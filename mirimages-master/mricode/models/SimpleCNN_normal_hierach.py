import tensorflow as tf
from tensorflow.keras.layers import Conv3D
from tensorflow import nn
from tensorflow.python.ops import nn_ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras.engine.base_layer import InputSpec
from tensorflow.python.keras.utils import conv_utils

from tensorflow.keras import Model

class MyDNN(Model):
  def __init__(self, cat_cols, num_cols):
    super(MyDNN, self).__init__()
    self.cat_cols = cat_cols
    self.num_cols = num_cols
    self.ac = tf.keras.layers.ReLU()
    self.maxpool = tf.keras.layers.MaxPool3D(pool_size=(2, 2, 2), data_format='channels_last')
    self.bn1 = tf.keras.layers.BatchNormalization()
    self.bn2 = tf.keras.layers.BatchNormalization()
    self.bn3 = tf.keras.layers.BatchNormalization()
    self.bn4 = tf.keras.layers.BatchNormalization()
    self.bn5 = tf.keras.layers.BatchNormalization()
    self.bn6 = tf.keras.layers.BatchNormalization()
    self.gapool = tf.keras.layers.GlobalAveragePooling3D()
    self.dense1 = tf.keras.layers.Dense(512, activation='relu')
    self.dense2 = tf.keras.layers.Dense(512, activation='relu')



    self.conv1 = tf.keras.layers.Conv3D(
        filters = 32,
        kernel_size = 3,
        padding='valid',
        data_format='channels_last'
    )
    self.conv2 = tf.keras.layers.Conv3D(
        filters = 64,
        kernel_size = 3,
        padding='valid',
        data_format='channels_last'
    )
   
    self.conv3 = tf.keras.layers.Conv3D(
        filters = 128,
        kernel_size = 3,
        padding='valid',
        data_format='channels_last'
    )
    self.conv4 = tf.keras.layers.Conv3D(
        filters = 256,
        kernel_size = 3,
        padding='valid',
        data_format='channels_last'
    )

    

    self.fc = {1:{},2:{}}
    for k in list(self.cat_cols.keys()):
      self.fc[1][k] = tf.keras.Sequential([
                                        tf.keras.layers.Dense(256, activation='relu'),
                                        tf.keras.layers.BatchNormalization(),
                                        tf.keras.layers.Dense(self.cat_cols[k], activation='softmax')
                                        ])
    for i in range(5):
      self.fc[1][self.num_cols[i]] = tf.keras.Sequential([
                                        tf.keras.layers.Dense(256, activation='relu'),
                                        tf.keras.layers.BatchNormalization(),
                                        tf.keras.layers.Dense(1)
                                        ])  
    
    for i in range(5,len(self.num_cols)):
      self.fc[2][self.num_cols[i]] = tf.keras.Sequential([
                                        tf.keras.layers.Dense(256, activation='relu'),
                                        tf.keras.layers.BatchNormalization(),
                                        tf.keras.layers.Dense(1)
                                        ]) 

    
    
  
  def call(self, x):
    

    out = {}
    
    #backbone-1
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.ac(x)
    x = self.maxpool(x)
    x = self.conv2(x)
    x = self.bn2(x)
    x = self.ac(x)
    
    #op-phenotypes
    out[1] = self.gapool(x)
    out[1] = self.dense1(out[1])
    out[1] = self.bn5(out[1])
    out[1] = self.ac(out[1])
    op = {}
    for i in list(self.fc[1].keys()):
      op[i] = self.fc[1][i](out[1])

    #backbone-2
    x = self.conv3(x)
    x = self.bn3(x)
    x = self.ac(x)
    x = self.maxpool(x)
    x = self.conv4(x)
    x = self.bn4(x)
    x = self.ac(x)    

    #op-intelligence
    out[2] = self.gapool(x)
    out[2] = self.dense2(out[2])
    out[2] = self.bn6(out[2])
    out[2] = self.ac(out[2])
    for i in list(self.fc[2].keys()):
      op[i] = self.fc[2][i](out[2])
    return op