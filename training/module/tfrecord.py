import tensorflow as tf
import module.data as data
import numpy as np

# The following functions can be used to convert a value to a type compatible
# with tf.Example.

def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  # if isinstance(value, type(tf.constant(0))):
  #   value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

# def _float_feature(value):
#   """Returns a float_list from a float / double."""
#   return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

# label from filename
def serialize_example_from_filename(file_path, new_filename, inverse_bool=False):
  """
  Creates a tf.Example message ready to be written to a file.
  """
  # New Filename: {SN}_{LOC}_{Comp}_{degree}_{capvalue}_{voltage}_{index}
  # Old Filename: {SN}_{LOC}_{degree}_{capvalue}_{voltage} 
  # Create a dictionary mapping the feature name to the tf.Example-compatible data type.
  splitted_fn = file_path.split('/')[-1].split('_')
  bytes_splitted_fn = [sn.encode() for sn in splitted_fn]
  with open(file_path, "rb") as f:
    image = f.read() # without decode_base64
  if new_filename is True:
    if inverse_bool is True:
      degree_value = str((int(splitted_fn[3])+180)%360).encode()
    else:
      degree_value = bytes_splitted_fn[3]
    feature = {
        'file_path': _bytes_feature(file_path.encode()),
        'Full_SN': _bytes_feature((splitted_fn[0]+'_'+splitted_fn[1]).encode()),
        'SN': _bytes_feature(bytes_splitted_fn[0]),
        'image': _bytes_feature(image),
        'component': _bytes_feature(bytes_splitted_fn[2]),
        'onehot_degree': _int64_feature(
          np.argmax(data.get_degree_label_from_filename(file_path, new_filename, inverse_bool).numpy())),
        'onehot_label': _int64_feature(
          np.argmax(data.get_label_from_filename(file_path, new_filename, inverse_bool).numpy())),
        'degree': _bytes_feature(degree_value),
        'capvalue': _bytes_feature(bytes_splitted_fn[4]),
        'voltage': _bytes_feature(bytes_splitted_fn[5]),
        'index': _bytes_feature(splitted_fn[6].split('.')[0].encode()),
    }
  else:
    if inverse_bool is True:
      degree_value = str((int(splitted_fn[2])+180)%360).encode()
    else:
      degree_value = bytes_splitted_fn[2]
    feature = {
        'file_path': _bytes_feature(file_path.encode()),
        'Full_SN': _bytes_feature((splitted_fn[0]+'_'+splitted_fn[1]).encode()),
        'SN': _bytes_feature(bytes_splitted_fn[0]),
        'image': _bytes_feature(image),
        'component': _bytes_feature(b'NA'),
        'onehot_degree': _int64_feature(
          np.argmax(data.get_degree_label_from_filename(file_path, new_filename, inverse_bool).numpy())),
        'onehot_label': _int64_feature(
          np.argmax(data.get_label_from_filename(file_path, new_filename, inverse_bool).numpy())),
        'degree': _bytes_feature(degree_value),
        'capvalue': _bytes_feature(bytes_splitted_fn[3]),
        'voltage': _bytes_feature(splitted_fn[4].split('.')[0].encode()),
        'index': _bytes_feature(b'NA'),
    }
  # Create a Features message using tf.train.Example.
  example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
  return example_proto.SerializeToString()

def write_tfrecord_file_from_filename(
  globbed_filepath, tfrecord_filepath, new_filename, inverse_bool=False):
  # Write the `tf.Example` observations to the file.
  with tf.io.TFRecordWriter(tfrecord_filepath) as writer:
    for filepath in globbed_filepath:
      example = serialize_example_from_filename(filepath, new_filename, inverse_bool)
      writer.write(example)
  print('Exported Successfully to '+tfrecord_filepath)

# label from folder
def serialize_example_from_folder(file_path, new_filename):
  """
  Creates a tf.Example message ready to be written to a file.
  """
  # New Filename: {SN}_{LOC}_{Comp}_{degree}_{capvalue}_{voltage}_{index}
  # Old Filename: {SN}_{LOC}_{degree}_{capvalue}_{voltage} 
  # Create a dictionary mapping the feature name to the tf.Example-compatible data type.
  splitted_fn = file_path.split('/')[-1].split('_')
  bytes_splitted_fn = [sn.encode() for sn in splitted_fn]
  with open(file_path, "rb") as f:
    image = f.read() # without decode_base64
  if new_filename is True:
    feature = {
        'file_path': _bytes_feature(file_path.encode()),
        'Full_SN': _bytes_feature((splitted_fn[0]+'_'+splitted_fn[1]).encode()),
        'SN': _bytes_feature(bytes_splitted_fn[0]),
        'image': _bytes_feature(image),
        'component': _bytes_feature(bytes_splitted_fn[2]),
        'onehot_degree': _int64_feature(
          np.argmax(data.get_degree_label_from_folder(file_path).numpy())),
        'onehot_label': _int64_feature(
          np.argmax(data.get_label_from_folder(file_path).numpy())),
        'degree': _bytes_feature(file_path.split('/')[-2].encode()),
        'capvalue': _bytes_feature(bytes_splitted_fn[4]),
        'voltage': _bytes_feature(bytes_splitted_fn[5]),
        'index': _bytes_feature(splitted_fn[6].split('.')[0].encode()),
    }
  else:
    feature = {
        'file_path': _bytes_feature(file_path.encode()),
        'Full_SN': _bytes_feature((splitted_fn[0]+'_'+splitted_fn[1]).encode()),
        'SN': _bytes_feature(bytes_splitted_fn[0]),
        'image': _bytes_feature(image),
        'component': _bytes_feature(b'NA'),
        'onehot_degree': _int64_feature(
          np.argmax(data.get_degree_label_from_folder(file_path).numpy())),
        'onehot_label': _int64_feature(
          np.argmax(data.get_label_from_folder(file_path).numpy())),
        'degree': _bytes_feature(file_path.split('/')[-2].encode()),
        'capvalue': _bytes_feature(bytes_splitted_fn[3]),
        'voltage': _bytes_feature(splitted_fn[4].split('.')[0].encode()),
        'index': _bytes_feature(b'NA'),
    }
  # Create a Features message using tf.train.Example.
  example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
  return example_proto.SerializeToString()

def write_tfrecord_file_from_folder(globbed_filepath, tfrecord_filepath, new_filename):
  # Write the `tf.Example` observations to the file.
  with tf.io.TFRecordWriter(tfrecord_filepath) as writer:
    for filepath in globbed_filepath:
      example = serialize_example_from_folder(filepath, new_filename)
      writer.write(example)
  print('Exported Successfully to '+tfrecord_filepath)

# Create a dictionary describing the features.
image_feature_description = {
    # 'file_path': tf.io.FixedLenFeature([], tf.string),
    # 'Full_SN': tf.io.FixedLenFeature([], tf.string),
    # 'SN': tf.io.FixedLenFeature([], tf.string),
    'image': tf.io.FixedLenFeature([], tf.string),
    'onehot_degree': tf.io.FixedLenFeature([], tf.int64),
    # 'onehot_label': tf.io.FixedLenFeature([], tf.int64),
    # 'degree': tf.io.FixedLenFeature([], tf.string),
    # 'capvalue': tf.io.FixedLenFeature([], tf.string),
    # 'voltage': tf.io.FixedLenFeature([], tf.string),
    # 'index': tf.io.FixedLenFeature([], tf.string),
}

def _parse_image_function(example_proto):
  # Parse the input tf.Example proto using the dictionary above.
  return tf.io.parse_single_example(example_proto, image_feature_description)

def read_tfrecord_file(tfrecord_filepath):
  ds = tf.data.TFRecordDataset(tfrecord_filepath)
  ds = ds.map(_parse_image_function, tf.data.experimental.AUTOTUNE)
  return ds

#######
def parse_image_degree(single_tfrecord, target_shape, ResizeMethod=tf.image.ResizeMethod.BILINEAR):
  img = data.decode_img(single_tfrecord['image'], target_shape, ResizeMethod)
  onehot_degree = tf.one_hot(single_tfrecord['onehot_degree'], data.DEGREE_NUM, dtype='int64')
  # onehot_degree = tf.keras.utils.to_categorical(single_tfrecord['onehot_degree'], num_classes=data.DEGREE_NUM)
  return img, onehot_degree

def parse_tfrecord_dataset(example, target_shape, ResizeMethod=tf.image.ResizeMethod.BILINEAR):
  return parse_image_degree(_parse_image_function(example), target_shape, ResizeMethod)

def before_sample_from_dataset(ds, buffer_size):
  return ds.shuffle(buffer_size).repeat()

# ratio should be between 0. and 1.
def mix_dataset_depending_on_ratio(a_ds, b_ds, b_ratio, bigger_swift=True):
  if b_ratio == 0.:
    mixed_ds = a_ds
  elif b_ratio == 1.:
    mixed_ds = b_ds
  else:
    a_ratio = 1 - b_ratio
    if bigger_swift == True:
      if len(list(a_ds)) > len(list(b_ds)):
        a_ds = a_ds.take(np.ceil(len(list(b_ds)) * (a_ratio / b_ratio)))
      else:
        b_ds = b_ds.take(np.ceil(len(list(a_ds)) * (b_ratio / a_ratio)))
    else:
      b_ds = b_ds.take(np.ceil(len(list(a_ds)) * (b_ratio / a_ratio)))
    # print(len(list(a_ds)), len(list(b_ds)), a_ratio, b_ratio)
    mixed_ds = tf.data.experimental.sample_from_datasets(
                      [a_ds, b_ds], weights=[a_ratio, b_ratio])
  return mixed_ds

def dataset_after_interleave_from_tfrecord(ds, target_shape,
                                batch_size, ResizeMethod=tf.image.ResizeMethod.BILINEAR,
                                cache=True, shuffle_buffer_size=20000, to_balance=True):
  ds = ds.map(lambda x: parse_tfrecord_dataset(x, target_shape, ResizeMethod), 
              num_parallel_calls=tf.data.experimental.AUTOTUNE)
  ds_step = int(np.ceil(len(list(ds)) / batch_size))
  if to_balance is True:
    # mixed_train_ds = mix_dataset_depending_on_ratio(ori_ds, generative_ds, 0.5)
    ng_train_ds = ds.filter(lambda image, label: 
                    tf.math.argmax(input = label)==1 or tf.math.argmax(input = label)==3)
    ok_train_ds = ds.filter(lambda image, label: 
                    tf.math.argmax(input = label)==0 or tf.math.argmax(input = label)==2)
    # ds = tf.data.experimental.sample_from_datasets(
    #   [ok_train_ds.take(len(list(ng_train_ds))), ng_train_ds], [0.5, 0.5])
    ds_step = int(np.ceil(len(list(ok_train_ds)) * 2 / batch_size))
    ds = tf.data.experimental.sample_from_datasets(
      [ok_train_ds.repeat(), ng_train_ds.repeat()], [0.5, 0.5])
    # print(len(list(ok_train_ds)), len(list(ng_train_ds)), len(list(ds)))
  ds = ds.shuffle(buffer_size=shuffle_buffer_size)
  if cache:
    if isinstance(cache, str):
        ds = ds.cache(cache)
    else:
        ds = ds.cache()
  ds = data.repeat_batch_prefetch(ds, batch_size)
  # ds = data.batch_prefetch(ds, batch_size)
  return (ds, ds_step)

def test_dataset_from_tfrecord(test_img_tfrecord_list, target_shape,
                                batch_size, ResizeMethod=tf.image.ResizeMethod.BILINEAR,
                                cache=True, shuffle_buffer_size=20000):
  test_ds = tf.data.Dataset.list_files(
              test_img_tfrecord_list).interleave(tf.data.TFRecordDataset)
  (test_ds, test_step) = dataset_after_interleave_from_tfrecord(
              test_ds, target_shape, batch_size, 
              ResizeMethod, cache, shuffle_buffer_size
  )
  return (test_ds, test_step)

def mix_train_dataset_with_generative_from_tfrecord(ori_img_tfrecord_list, stylized_img_tfrecord_list, 
                                        dcgan_img_tfrecord_list, valid_img_tfrecord_list, 
                                        target_shape, dcgan_ratio, generative_ratio, 
                                        batch_size, ResizeMethod=tf.image.ResizeMethod.BILINEAR, 
                                        to_balance=True, cache=True, shuffle_buffer_size=20000):
  ori_ds = tf.data.Dataset.list_files(
              ori_img_tfrecord_list).interleave(tf.data.TFRecordDataset)
  if generative_ratio != 0.:
    stylized_ds = tf.data.Dataset.list_files(
                stylized_img_tfrecord_list, shuffle=True).interleave(tf.data.TFRecordDataset)            
    dcgan_ds = tf.data.Dataset.list_files(
                dcgan_img_tfrecord_list, shuffle=True).interleave(tf.data.TFRecordDataset)
    generative_ds = mix_dataset_depending_on_ratio(stylized_ds, dcgan_ds, dcgan_ratio)
    # the instance below is ori_ds is larger than generative_ds
    mixed_train_ds = mix_dataset_depending_on_ratio(ori_ds, generative_ds, generative_ratio, False)
    # print(len(list(ori_ds)), len(list(generative_ds)), len(list(mixed_train_ds)))
  else:
    mixed_train_ds = ori_ds

  if valid_img_tfrecord_list != []:
    valid_ds = tf.data.Dataset.list_files(
                valid_img_tfrecord_list).interleave(tf.data.TFRecordDataset)
  else:
    mixed_train_ds, valid_ds = data.split_dataset(mixed_train_ds, 0.2)
  (valid_ds, valid_step) = dataset_after_interleave_from_tfrecord(
              valid_ds, target_shape, batch_size, 
              ResizeMethod, cache, shuffle_buffer_size
    )
  (mixed_train_ds, train_step) = dataset_after_interleave_from_tfrecord(
              mixed_train_ds, target_shape, batch_size, 
              ResizeMethod, cache, shuffle_buffer_size, to_balance
  )
  return (mixed_train_ds, valid_ds, train_step, valid_step)

####### image degree label test ########
def parse_image_degree_label(single_tfrecord, target_shape, ResizeMethod=tf.image.ResizeMethod.BILINEAR):
  img = data.decode_img(single_tfrecord['image'], target_shape, ResizeMethod)
  onehot_label = tf.one_hot(single_tfrecord['onehot_label'], data.LABEL_NUM, dtype='int64')
  onehot_degree = tf.one_hot(single_tfrecord['onehot_degree'], data.DEGREE_NUM, dtype='int64')
  # onehot_degree = tf.keras.utils.to_categorical(single_tfrecord['onehot_degree'], num_classes=data.DEGREE_NUM)
  return img, onehot_degree, onehot_label

def parse_tfrecord_dataset_with_label(example, target_shape, ResizeMethod=tf.image.ResizeMethod.BILINEAR):
  return parse_image_degree_label(_parse_image_function(example), target_shape, ResizeMethod)

def dataset_with_label_to_ready(tfrecord_filepath_list, target_shape, batch_size, 
                                        ResizeMethod=tf.image.ResizeMethod.BILINEAR, 
                                        cache=True, shuffle_buffer_size=20000):
  ds = tf.data.Dataset.list_files(
              test_img_tfrecord_list).interleave(tf.data.TFRecordDataset)
  ds = ds.map(lambda x: parse_tfrecord_dataset(x, target_shape, ResizeMethod), 
                                      num_parallel_calls=tf.data.experimental.AUTOTUNE)
  if cache:
      if isinstance(cache, str):
          ds = ds.cache(cache)
      else:
          ds = ds.cache()
  ds = ds.shuffle(buffer_size=shuffle_buffer_size)
  return ds

def from_tfrecord_to_splitted_dataset(tfrecord_filepath, target_shape,  
                                        valid_ratio, batch_size, 
                                        cache=True, shuffle_buffer_size=20000):
  shuffled_ds = dataset_from_tfrecord_filepath_to_shuffle(
      tfrecord_filepath, target_shape, cache, shuffle_buffer_size)
  train_ds, valid_ds = data.split_dataset(shuffled_ds, valid_ratio)
  train_step = int(np.ceil(len(list(train_ds)) / batch_size))
  valid_step = int(np.ceil(len(list(valid_ds)) / batch_size))
  train_ds = data.repeat_batch_prefetch(train_ds, batch_size)
  valid_ds = data.repeat_batch_prefetch(valid_ds, batch_size)
  return train_ds, valid_ds, train_step, valid_step

#########