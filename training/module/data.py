import tensorflow as tf
import glob
import math

DEGREE_CLASS_LIST = ["0", "180", "270", "90"]
DEGREE_INDEX_LOOKUP = tf.constant(list(range(len(DEGREE_CLASS_LIST))), dtype=tf.int64)
DEGREE_NUM = len(DEGREE_CLASS_LIST)
lookup_table_degree = tf.lookup.StaticHashTable(
    tf.lookup.KeyValueTensorInitializer(DEGREE_CLASS_LIST, DEGREE_INDEX_LOOKUP), DEGREE_NUM)
LABEL_CLASS_LIST = ["OK", "NG"]
DEGREE_TO_LABEL_LIST = ["OK", "NG", "OK", "NG"]
LABEL_INDEX_LOOKUP = tf.constant(list(range(len(LABEL_CLASS_LIST))), dtype=tf.int64)
LABEL_NUM = len(LABEL_CLASS_LIST)
lookup_table_degree_to_label_value = tf.lookup.StaticHashTable(
    tf.lookup.KeyValueTensorInitializer(DEGREE_CLASS_LIST, DEGREE_TO_LABEL_LIST), "")
lookup_table_label_index = tf.lookup.StaticHashTable(
    tf.lookup.KeyValueTensorInitializer(LABEL_CLASS_LIST, LABEL_INDEX_LOOKUP), LABEL_NUM)

def decode_img(byte_string_img, target_shape, 
                ResizeMethod=tf.image.ResizeMethod.BILINEAR):
    img = tf.io.decode_png(byte_string_img, channels=target_shape[-1])
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, target_shape[0:2], method=ResizeMethod)
    return img

def split_dataset(dataset, valid_ratio):
    ds_size = len(list(dataset))
    valid_size = int(math.ceil(ds_size * valid_ratio))
    valid_ds = dataset.take(valid_size)
    train_ds = dataset.skip(valid_size)
    return train_ds, valid_ds

def batch_prefetch(ds, BATCH_SIZE):
    ds = ds.batch(BATCH_SIZE)
    ds = ds.prefetch(2)
    # ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return ds

def repeat_batch_prefetch(ds, BATCH_SIZE):
    ds = ds.repeat()
    ds = ds.batch(BATCH_SIZE)
    ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return ds

# label from filename
def get_degree_label_from_filename(file_path, new_filename, inverse_bool):
    filename = tf.strings.split(file_path, '/')[-1]
    if new_filename is True:
        degree = tf.strings.split(filename, '_')[-4]
    else:
        degree = tf.strings.split(filename, '_')[-3]
    if inverse_bool is True:
        degree = tf.strings.as_string((int(degree) + 180) % 360)
    onehot_degree = tf.one_hot(lookup_table_degree.lookup(degree), DEGREE_NUM)
    return onehot_degree

def get_label_from_filename(file_path, new_filename, inverse_bool):
    filename = tf.strings.split(file_path, '/')[-1]
    if new_filename is True:
        degree = tf.strings.split(filename, '_')[-4]
    else:
        degree = tf.strings.split(filename, '_')[-3]
    if inverse_bool is True:
        degree = tf.strings.as_string((int(degree) + 180) % 360)
    onehot_label = tf.one_hot(lookup_table_label_index.lookup(
        lookup_table_degree_to_label_value.lookup(degree)), LABEL_NUM)
    return onehot_label

def process_image_degree_from_filename(file_path, target_shape, new_filename, inverse_bool): 
    label = get_degree_label_from_filename(file_path, new_filename, inverse_bool)
    byte_string_img = tf.io.read_file(file_path)
    img = decode_img(byte_string_img, target_shape)
    return img, label

def dataset_from_filepath_to_shuffle(filepath_with_stars, target_shape, new_filename, inverse_bool=False,
                                        cache=True, shuffle_buffer_size=20000):
    list_filepath_ds = tf.data.Dataset.list_files(filepath_with_stars)
    ds = list_filepath_ds.map(lambda x: process_image_degree_from_filename(x, target_shape, new_filename, inverse_bool), 
                                        num_parallel_calls=tf.data.experimental.AUTOTUNE)
    if cache:
        if isinstance(cache, str):
            ds = ds.cache(cache)
        else:
            ds = ds.cache()
    ds = ds.shuffle(buffer_size=shuffle_buffer_size)
    return ds

def from_filepath_to_splitted_dataset(filepath_with_stars, target_shape, new_filename,  
                                        valid_ratio, batch_size, inverse_bool=False, 
                                        cache=True, shuffle_buffer_size=20000):
    shuffled_ds = dataset_from_filepath_to_shuffle(
        filepath_with_stars, target_shape, new_filename, inverse_bool, cache, shuffle_buffer_size)
    train_ds, valid_ds = split_dataset(shuffled_ds, valid_ratio)
    train_step = int(math.ceil(len(list(train_ds)) / batch_size))
    valid_step = int(math.ceil(len(list(valid_ds)) / batch_size))
    train_ds = repeat_batch_prefetch(train_ds, batch_size)
    valid_ds = repeat_batch_prefetch(valid_ds, batch_size)
    return train_ds, valid_ds, train_step, valid_step

# label from folder
def get_degree_label_from_folder(file_path):
    degree = tf.strings.split(file_path, '/')[-2]
    onehot_degree = tf.one_hot(lookup_table_degree.lookup(degree), DEGREE_NUM)
    return onehot_degree

def get_label_from_folder(file_path):
    degree = tf.strings.split(file_path, '/')[-2]
    onehot_label = tf.one_hot(lookup_table_label_index.lookup(
        lookup_table_degree_to_label_value.lookup(degree)), LABEL_NUM)
    return onehot_label


def process_image_degree_from_folder(file_path, target_shape): 
    label = get_degree_label_from_folder(file_path)
    byte_string_img = tf.io.read_file(file_path)
    img = decode_img(byte_string_img, target_shape)
    return img, label

def label_from_folder_to_shuffle(filepath_with_stars, target_shape,
                                        cache=True, shuffle_buffer_size=20000):
    list_filepath_ds = tf.data.Dataset.list_files(filepath_with_stars)
    ds = list_filepath_ds.map(lambda x: process_image_degree_from_folder(x, target_shape), 
                                        num_parallel_calls=tf.data.experimental.AUTOTUNE)
    if cache:
        if isinstance(cache, str):
            ds = ds.cache(cache)
        else:
            ds = ds.cache()
    ds = ds.shuffle(buffer_size=shuffle_buffer_size)
    return ds

def label_from_folder_to_splitted_dataset(filepath_with_stars, target_shape,  
                                        valid_ratio, batch_size, 
                                        cache=True, shuffle_buffer_size=20000):
    shuffled_ds = label_from_folder_to_shuffle(
        filepath_with_stars, target_shape, cache, shuffle_buffer_size)
    train_ds, valid_ds = split_dataset(shuffled_ds, valid_ratio)
    train_step = int(math.ceil(len(list(train_ds)) / batch_size))
    valid_step = int(math.ceil(len(list(valid_ds)) / batch_size))
    train_ds = repeat_batch_prefetch(train_ds, batch_size)
    valid_ds = repeat_batch_prefetch(valid_ds, batch_size)
    return train_ds, valid_ds, train_step, valid_step