import tensorflow as tf
from tensorflow.keras.layers import Conv3D
from tensorflow import nn
from tensorflow.python.ops import nn_ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras.engine.base_layer import InputSpec
from tensorflow.python.keras.utils import conv_utils

from tensorflow.keras import Model

class CrossStitchUnit(tf.keras.layers.Layer):
  """
    CrossStitch Unit
  """
  def __init__(self, num_tasks):

    super(CrossStitchUnit, self).__init__()
    self.num_outputs = num_tasks
    self.kernel = self.add_weight("weight",
                                    shape=[self.num_outputs,self.num_outputs],
                                    initializer='identity',
                                    trainable=True)
    
  def call(self, ip):
    op_list = []
    for task_i in range(self.num_outputs):
      op = 0
      for task_j in range(self.num_outputs):
        task_ij_wt =  self.kernel[task_i, task_j]
        op += tf.math.scalar_mul(task_ij_wt, ip[task_j])
      op_list.append(op)
    return tf.stack(op_list, axis=0)



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
    
    self.fc = {}
    for k in list(self.cat_cols.keys()):
      self.fc[k] = tf.keras.Sequential([
                                        tf.keras.layers.Dense(256, activation='relu'),
                                        tf.keras.layers.BatchNormalization(),
                                        tf.keras.layers.Dense(1024, activation='relu'),
                                        tf.keras.layers.BatchNormalization(),
                                        tf.keras.layers.Dense(self.cat_cols[k], activation='softmax')
                                        ])

    for i in range(len(self.num_cols)):
      self.fc[self.num_cols[i]] = tf.keras.Sequential([
                                        tf.keras.layers.Dense(256, activation='relu'),
                                        tf.keras.layers.BatchNormalization(),
                                        tf.keras.layers.Dense(1024, activation='relu'),
                                        tf.keras.layers.BatchNormalization(),
                                        tf.keras.layers.Dense(1)
                                        ])

    
    self.cs1 = CrossStitchUnit(len(list(self.fc.keys())))
    self.cs2 = CrossStitchUnit(len(list(self.fc.keys())))
    self.cs3 = CrossStitchUnit(len(list(self.fc.keys())))


  
  def call(self, x):
    
    def feeder_func(dic_d,list_l):
      i = 0
      for k in list(dic_d.keys()):
        dic_d[k] = list_l[i]
        i+=1
      return dic_d,[]

    out = {}
    vals = []
    for k in list(self.fc.keys()):

      out[k] = self.conv1(x)
      out[k] = self.bn1(out[k])
      out[k] = self.ac(out[k])
      out[k] = self.maxpool(out[k])
      vals.append(out[k])
    vals = tf.unstack(self.cs1(vals), axis=0)
    out,vals = feeder_func(out,vals)
    
    for k in list(self.fc.keys()):
      out[k] = self.conv2(out[k])
      out[k] = self.bn2(out[k])
      out[k] = self.ac(out[k])
      out[k] = self.maxpool(out[k])
      vals.append(out[k])
    vals = tf.unstack(self.cs2(vals), axis=0)
    out,vals = feeder_func(out,vals)

    for k in list(self.fc.keys()):
      out[k] = self.conv3(out[k])
      out[k] = self.bn3(out[k])
      out[k] = self.ac(out[k])
      out[k] = self.maxpool(out[k])
      vals.append(out[k])
    vals = tf.unstack(self.cs3(vals), axis=0)
    out,vals = feeder_func(out,vals)

    for k in list(self.fc.keys()):
      out[k] = self.conv4(out[k])
      out[k] = self.bn4(out[k])
      out[k] = self.ac(out[k])
      out[k] = tf.keras.layers.GlobalAveragePooling3D()(out[k])
      out[k] = self.fc[k](out[k])
    
    return out