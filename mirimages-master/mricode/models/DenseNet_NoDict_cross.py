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



class MyDenseNet(Model):
  def __init__(self, cat_cols, num_cols):
    super(MyDenseNet, self).__init__()
    self.cat_cols = cat_cols
    self.num_cols = num_cols
    

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
    self.cs1 = CrossStitchUnit(2)
    self.cs2 = CrossStitchUnit(2)
    self.cs3 = CrossStitchUnit(2)


  def call(self, x):

    def feeder_func(dic_d,list_l):
      i = 0
      for k in list(dic_d.keys()):
        dic_d[k] = list_l[i]
        i+=1
      return dic_d,[]


    out = {}
    vals = []
    for k in [1,2]:

      out[k] = self.conv1(x)
      # print("initial shape: ", x.shape)

      ## dense block 1 ##
      num_blocks = 4
      layers_concat = list()
      layers_concat.append(out[k])
    
      out[k] = self.bottleneck1[0](out[k])
      # print("first bottleneck shape: ", x.shape)
      layers_concat.append(out[k])
      
      for d in range(1, num_blocks):
        out[k] = tf.concat(layers_concat, axis=4)
        # print("first concat: ", x.shape)
        out[k] = self.bottleneck1[d](out[k])
        layers_concat.append(out[k])
      out[k] = tf.concat(layers_concat, axis=4)
      ## end dense block 1 ##

      out[k] = self.transition1(out[k])
      # print("first transition shape: ", x.shape)
      vals.append(out[k])
    
    vals = tf.unstack(self.cs1(vals), axis=0)
    out,vals = feeder_func(out,vals)


    for k in [1,2]:

      ## dense block 2 ##
      num_blocks = 4
      layers_concat = list()
      layers_concat.append(out[k])
    
      out[k] = self.bottleneck2[0](out[k])
      layers_concat.append(out[k])
      
      for d in range(1, num_blocks):
        out[k] = tf.concat(layers_concat, axis=4)
        out[k] = self.bottleneck2[d](out[k])
        layers_concat.append(out[k])
      out[k] = tf.concat(layers_concat, axis=4)
      ## end dense block 2 ##

      out[k] = self.transition2(out[k])
      vals.append(out[k])

    vals = tf.unstack(self.cs2(vals), axis=0)
    out,vals = feeder_func(out,vals)

    for k in [1,2]:

      ## dense block 3 ##
      num_blocks = 4
      layers_concat = list()
      layers_concat.append(out[k])
    
      out[k] = self.bottleneck3[0](out[k])
      layers_concat.append(out[k])
      
      for d in range(1, num_blocks):
        out[k] = tf.concat(layers_concat, axis=4)
        out[k] = self.bottleneck3[d](out[k])
        layers_concat.append(out[k])
      out[k] = tf.concat(layers_concat, axis=4)
      ## end dense block 3 ##

      out[k] = self.transition3(out[k])
      vals.append(out[k])

    vals = tf.unstack(self.cs3(vals), axis=0)
    out,vals = feeder_func(out,vals)

    op = {}
    for k in [1,2]:

      # END PART
      out[k] = self.bn1(out[k])
      out[k] = self.ac(out[k])
      out[k] = tf.keras.layers.GlobalAveragePooling3D()(out[k])
      out[k] = tf.keras.layers.Flatten()(out[k])
      for i in list(self.fc[k].keys()):
        op[i] = self.fc[k][i](out[k])

    return op