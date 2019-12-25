import tensorflow as tf
import pandas as pd
import numpy as np
import os
import nibabel as nib

def get_filenames(sub, all_files):
    return([x for x in all_files if sub in x])

def data_generator(subjectkeys,
                   all_files,
                   batch_size,
                   is_validation_data=False, 
                   input_shape=(128,128,128)):
    # Get total number of samples in the data
    n = len(subjectkeys)
    nb_batches = int(np.ceil(n/batch_size))

    # Get a numpy array of all the indices of the input data
    indices = np.arange(n)
    
    while True:
        if not is_validation_data:
            # shuffle indices for the training data
            np.random.shuffle(indices)
            
        for i in range(nb_batches):
            # get the next batch 
            next_batch_indices = indices[i*batch_size:(i+1)*batch_size]
            nb_examples = len(next_batch_indices)
            
            # Define two numpy arrays for containing batch data and labels
            t1 = np.zeros((nb_examples, 
                           input_shape[0], 
                           input_shape[1],
                           input_shape[2],
                           1), 
                          dtype=np.float32)
            t2 = np.zeros((nb_examples, 
                           input_shape[0], 
                           input_shape[1],
                           input_shape[2],
                           1), 
                          dtype=np.float32)
            subjectkey = []
            
            # process the next batch
            for j, idx in enumerate(next_batch_indices):
                subjectkey.append(subjectkeys[idx])
                subject_files = get_filenames(subjectkeys[idx], all_files)
                t1_name = [x for x in subject_files if '_T1.nii.gz' in x][0]
                t2_name = [x for x in subject_files if '_T2.nii.gz' in x][0]
                
                t1_img = nib.load(t1_name)
                t1_img = np.array(t1_img.dataobj)
                t2_img = nib.load(t2_name)
                t2_img = np.array(t2_img.dataobj)
                t1[j] = np.expand_dims(t1_img, axis=-1)
                t2[j] = np.expand_dims(t2_img, axis=-1)

            yield {'t1': tf.convert_to_tensor(t1), 't2': tf.convert_to_tensor(t2), 'subjectid': tf.convert_to_tensor(subjectkey)}

def createPath(path):
    """createPath
    
    Function creates a path, if the path does not exist
    
    Args:
        path (string): Path to create
    """
    if not os.path.exists(path):
        os.makedirs(path)

def log_textfile(filename, text):
    """
    Function log_to_textfile
    
    Appends a text to a file (logs)
    
    Args:
        filename (str): Filename of logfile
        text (str): New information to log (append)
    
    Return:
    
    """
    print(text)
    f = open(filename, "a")
    f.write(str(text) + str('\n'))
    f.close()

def copy_colab(path_tfrecords, path_csv, sample_size, filenames={'train': 'intell_train.csv', 'val': 'intell_valid.csv', 'test': 'intell_test.csv'}):
  os.system('mkdir /content/files/')
  os.system('cp ' + path_tfrecords.replace(' ', '\ ') + 't1t2_train_' + str(sample_size) + '_v4.tfrecords' + ' /content/files/')
  os.system('cp ' + path_tfrecords.replace(' ', '\ ') + 't1t2_val_' + str(sample_size) + '_v4.tfrecords' + ' /content/files/')
  os.system('cp ' + path_tfrecords.replace(' ', '\ ') + 't1t2_test_' + str(sample_size) + '_v4.tfrecords' + ' /content/files/')
  os.system('cp ' + path_csv.replace(' ', '\ ') + filenames['train'] + ' /content/files/')
  os.system('cp ' + path_csv.replace(' ', '\ ') + filenames['val'] + ' /content/files/')
  os.system('cp ' + path_csv.replace(' ', '\ ') + filenames['test'] + ' /content/files/')

def return_iter(path, sample_size, batch_size=8, onlyt1=False, bl_cropped=False, dim=64):
  # Some definitions
  if onlyt1:
    read_features = {
      't1': tf.io.FixedLenFeature([], dtype=tf.string),
      't2': tf.io.FixedLenFeature([], dtype=tf.string),
      'subjectid': tf.io.FixedLenFeature([], dtype=tf.string)
    }
  else:
    read_features = {
      't1': tf.io.FixedLenFeature([], dtype=tf.string),
      't2': tf.io.FixedLenFeature([], dtype=tf.string),
      'ad': tf.io.FixedLenFeature([], dtype=tf.string),
      'fa': tf.io.FixedLenFeature([], dtype=tf.string),
      'md': tf.io.FixedLenFeature([], dtype=tf.string),
      'rd': tf.io.FixedLenFeature([], dtype=tf.string),
      'subjectid': tf.io.FixedLenFeature([], dtype=tf.string)
    }
  def _parse_(serialized_example, decoder = np.vectorize(lambda x: x.decode('UTF-8')), onlyt1=False, bl_cropped=False, dim=64):
    example = tf.io.parse_single_example(serialized_example, read_features)
    subjectid = example['subjectid']
    if not(onlyt1):
      if bl_cropped:
        t1 = tf.expand_dims(tf.reshape(tf.io.decode_raw(example['t1'], tf.float64), (dim,dim,dim)), axis=-1)
        t2 = tf.expand_dims(tf.reshape(tf.io.decode_raw(example['t2'], tf.float32), (dim,dim,dim)), axis=-1)
        ad = tf.expand_dims(tf.reshape(tf.io.decode_raw(example['ad'], tf.float32), (dim,dim,dim)), axis=-1)
        fa = tf.expand_dims(tf.reshape(tf.io.decode_raw(example['fa'], tf.float32), (dim,dim,dim)), axis=-1)
        md = tf.expand_dims(tf.reshape(tf.io.decode_raw(example['md'], tf.float32), (dim,dim,dim)), axis=-1)
        rd = tf.expand_dims(tf.reshape(tf.io.decode_raw(example['rd'], tf.float32), (dim,dim,dim)), axis=-1)
        return ({'t1': t1, 't2': t2, 'ad': ad, 'fa':fa, 'md': md, 'rd': rd,'subjectid': subjectid})
      else:
        t1 = tf.expand_dims(tf.reshape(tf.io.decode_raw(example['t1'], tf.int8), (dim,dim,dim)), axis=-1)
        t2 = tf.expand_dims(tf.reshape(tf.io.decode_raw(example['t2'], tf.float32), (dim,dim,dim)), axis=-1)
        ad = tf.expand_dims(tf.reshape(tf.io.decode_raw(example['ad'], tf.float32), (dim,dim,dim)), axis=-1)
        fa = tf.expand_dims(tf.reshape(tf.io.decode_raw(example['fa'], tf.float32), (dim,dim,dim)), axis=-1)
        md = tf.expand_dims(tf.reshape(tf.io.decode_raw(example['md'], tf.float32), (dim,dim,dim)), axis=-1)
        rd = tf.expand_dims(tf.reshape(tf.io.decode_raw(example['rd'], tf.float32), (dim,dim,dim)), axis=-1)
        return ({'t1': t1, 't2': t2, 'ad': ad, 'fa':fa, 'md': md, 'rd': rd,'subjectid': subjectid})
    else:
      if bl_cropped:
        t1 = tf.expand_dims(tf.reshape(tf.io.decode_raw(example['t1'], tf.float64), (dim,dim,dim)), axis=-1)
        t2 = tf.expand_dims(tf.reshape(tf.io.decode_raw(example['t2'], tf.float32), (dim,dim,dim)), axis=-1)
        return ({'t1': t1, 't2': t2, 'subjectid': subjectid})
      else:
        t1 = tf.expand_dims(tf.reshape(tf.io.decode_raw(example['t1'], tf.int8), (dim,dim,dim)), axis=-1)
        t2 = tf.expand_dims(tf.reshape(tf.io.decode_raw(example['t2'], tf.float32), (dim,dim,dim)), axis=-1)
        return ({'t1': t1, 't2': t2, 'subjectid': subjectid})
  
  train_ds = tf.data.TFRecordDataset(path +'t1t2_train_' + str(sample_size) + '_v4.tfrecords', num_parallel_reads=32)
  val_ds = tf.data.TFRecordDataset(path + 't1t2_val_' + str(sample_size) + '_v4.tfrecords', num_parallel_reads=32)
  test_ds = tf.data.TFRecordDataset(path + 't1t2_test_' + str(sample_size) + '_v4.tfrecords', num_parallel_reads=32)
  train_iter = train_ds.map(lambda x:_parse_(x, onlyt1=onlyt1, bl_cropped=bl_cropped, dim=dim), num_parallel_calls=32).shuffle(32, reshuffle_each_iteration=True).batch(batch_size)
  val_iter = val_ds.map(lambda x:_parse_(x, onlyt1=onlyt1, bl_cropped=bl_cropped, dim=dim), num_parallel_calls=32).batch(batch_size)
  test_iter = test_ds.map(lambda x:_parse_(x, onlyt1=onlyt1, bl_cropped=bl_cropped, dim=dim), num_parallel_calls=32).batch(batch_size)
  return train_iter, val_iter, test_iter

def return_csv(path, filenames={'train': 'intell_train.csv', 'val': 'intell_valid.csv', 'test': 'intell_test.csv'}, fluid = False):
  train_df = pd.read_csv(path + filenames['train'])
  val_df = pd.read_csv(path + filenames['val'])
  test_df = pd.read_csv(path + filenames['test'])
  norm = None
  if fluid:
    train_df.columns = ['subjectkey', 'fluid_res', 'fluid']
    val_df.columns = ['subjectkey', 'fluid_res', 'fluid']
    test_df.columns = ['subjectkey', 'fluid_res', 'fluid']
  train_df['subjectkey'] = train_df['subjectkey'].str.replace('_', '')
  val_df['subjectkey'] = val_df['subjectkey'].str.replace('_', '')
  test_df['subjectkey'] = test_df['subjectkey'].str.replace('_', '')
  if not(fluid):
    for df in [train_df, val_df, test_df]:
      df['race.ethnicity'] = df['race.ethnicity'] - 1
      df['married'] = df['married'] - 1
      df['high.educ_group'] = 0
      df.loc[(train_df['high.educ']>=11) & (df['high.educ']<=12),'high.educ_group'] = 1
      df.loc[(train_df['high.educ']>=13) & (df['high.educ']<=13),'high.educ_group'] = 2
      counter = 3
      for i in range(14,22):
        df.loc[(df['high.educ']>=i) & (df['high.educ']<=i),'high.educ_group'] = counter
      df['income_group'] = 0
      counter = 1
      for i in range(4,11):
        df.loc[(df['income']>=i) & (df['income']<=i),'income_group'] = counter
        counter += 1
    norm = {}
    for col in ['BMI', 'age', 'vol', 'weight', 'height', 'nihtbx_fluidcomp_uncorrected', 'nihtbx_cryst_uncorrected', 
                'nihtbx_pattern_uncorrected', 'nihtbx_picture_uncorrected', 
                'nihtbx_list_uncorrected', 'nihtbx_flanker_uncorrected',
                'nihtbx_picvocab_uncorrected', 'nihtbx_cardsort_uncorrected',
                'nihtbx_totalcomp_uncorrected', 'nihtbx_reading_uncorrected']:
      mean = train_df[col].mean()
      std = train_df[col].std()
      train_df[col + '_norm'] = (train_df[col]-mean)/std
      val_df[col + '_norm'] = (val_df[col]-mean)/std
      test_df[col + '_norm'] = (test_df[col]-mean)/std
      norm[col] = {'mean': mean, 'std': std}
  return train_df, val_df, test_df, norm