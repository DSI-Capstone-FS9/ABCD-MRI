import tensorflow as tf
from tensorflow.keras.layers import Conv3D
from tensorflow import nn
from tensorflow.python.ops import nn_ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras.engine.base_layer import InputSpec
from tensorflow.python.keras.utils import conv_utils

from tensorflow.keras import Model

class OwnAutofocus3D(Model):
    """
    Implements the Autofocus layer as described in
    https://link.springer.com/content/pdf/10.1007%2F978-3-030-00931-1_69.pdf
    """
    def __init__(self, dilations, filters, kernel_size, input_shape, activation=None,
                 attention_kernel_size=3, attention_filters=None,
                 attention_activation=tf.nn.relu, **kwargs):
        """
        """
        super(OwnAutofocus3D, self).__init__()
        # Init base tf 3D Conv class
        self.kernel_size = kernel_size
        self.ac = tf.keras.layers.ReLU()
        self.input_channels = input_shape[4]
        self.num_dilations = len(dilations) + 1
        self.conv3d1 = Conv3D(filters, kernel_size, padding='same')
        self.conv3d1.build(tensor_shape.TensorShape(input_shape))
        
        self.dil = []
        for d in dilations:
          self.dil.append([1, d, d, d, 1])
        self.bn = []
        for i in range(self.num_dilations):
          self.bn.append(tf.keras.layers.BatchNormalization(axis=-1))
        self.convatt1 = Conv3D(max(int(self.input_channels/2),1), kernel_size=attention_kernel_size, padding='same')
        self.convatt2 = Conv3D(self.num_dilations, kernel_size=1, padding='same')
        self.convatt1.build(tensor_shape.TensorShape(input_shape))
        self.convatt2.build(tensor_shape.TensorShape((input_shape[0], input_shape[1], input_shape[2], input_shape[3], max(int(self.input_channels/2),1))))

    def _weight_variable(self, name, shape):
      return tf.compat.v1.get_variable(name, shape, DTYPE, tf.compat.v1.truncated_normal_initializer(stddev=0.1))
    
    def _bias_variable(self, name, shape):
      return tf.compat.v1.get_variable(name, shape, DTYPE, tf.compat.v1.constant_initializer(0.1, dtype=DTYPE))

    def call(self, x, **kwargs):
        """
        Build computation graph
        Applies a convolution operation to 'x' for each dilation specified
        in 'self.dilations' with shared kernel and bias weights
        Processes 'x' through the attention layer mechanism as
        att = softmax(conv3D(relu(conv3D(x))))
        """
        att = self.ac(self.convatt1(x))
        att = self.convatt2(att)
        att = tf.keras.layers.Softmax()(att)

        out = []
        x1 = self.bn[0](self.conv3d1(x)) * tf.expand_dims(att[:,:,:,:,0], axis=-1)
        counter = 1
        for d in self.dil:
          p = int(((self.kernel_size-1)*(d[1]-1)+self.kernel_size)/2)
          x_pad = tf.pad(x, [[0,0], [p, p], [p, p], [p, p], [0, 0]], "CONSTANT")
          x1 += self.bn[counter](
              tf.nn.conv3d(x_pad, self.conv3d1.weights[0], [1, 1, 1, 1, 1], dilations=d, padding='VALID')+self.conv3d1.weights[1]
              ) * tf.expand_dims(att[:,:,:,:,counter], axis=-1)
          counter += 1
        x1 =  self.ac(x1)
        return(x1)

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
                                                 OwnAutofocus3D([2,4,6], filter_num2, 3, (8,64,64,64,filter_num1))
                                                 ]))
      
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
                                                OwnAutofocus3D([2,4,6], filter_num2, 3, (8,32,32,32,filter_num1))
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
                                                OwnAutofocus3D([2,4,6], 12, 3, (8,16,16,16,24))
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