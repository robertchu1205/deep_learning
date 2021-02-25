import tensorflow as tf
import module.conv_training as conv_training
# import module.transfer_training as transfer_training
import random

def to_ds_degree(degree):
    ds = tf.data.experimental.SqlDataset(
    "sqlite", "/data/aoi-wzs-p1-dip-fa-nvidia/training/p1-dip-metadata.db", 
    f"""select path, degree from metadata 
    where degree = '{degree}' and 
    component_class = 'label' and 
    label = 'OK'
    """, (tf.string, tf.string))
    return ds

def to_ds_label(label):
    ds = tf.data.experimental.SqlDataset(
    "sqlite", "/data/aoi-wzs-p1-dip-fa-nvidia/training/p1-dip-metadata.db", 
    f"""select path, label from metadata 
    where label = '{label}' and 
    component_class = 'label'
    """, (tf.string, tf.string))
    return ds

# def to_ds_comp_label(label, component):
#     ds = tf.data.experimental.SqlDataset(
#             "sqlite", "/data/aoi-wzs-p1-dip-fa-nvidia/training/p1-dip-metadata.db", 
#             f"""select path, label from metadata 
#             where label = '{label}' and 
#             component_class = '{component}' 
#             """, (tf.string, tf.string))
#     return ds

tfrecords = [
    # '/data/aoi-wzs-p1-dip-fa-nvidia/label_heatsink_screw/tfrecord/other_comps_random_1w.tfrecord',
    # '/data/aoi-wzs-p1-dip-fa-nvidia/label_heatsink_screw/tfrecord/stylized_screw_heatsink_before_20200503.tfrecord',
    '/data/aoi-wzs-p1-dip-fa-nvidia/label_heatsink_screw/tfrecord/LabelOrientationGenerateImage/NG.tfrecord',
    '/data/aoi-wzs-p1-dip-fa-nvidia/label_heatsink_screw/tfrecord/LabelOrientationGenerateImage/000.tfrecord',
    '/data/aoi-wzs-p1-dip-fa-nvidia/label_heatsink_screw/tfrecord/LabelOrientationGenerateImage/090.tfrecord',
    '/data/aoi-wzs-p1-dip-fa-nvidia/label_heatsink_screw/tfrecord/LabelOrientationGenerateImage/180.tfrecord',
    '/data/aoi-wzs-p1-dip-fa-nvidia/label_heatsink_screw/tfrecord/LabelOrientationGenerateImage/270.tfrecord',
]

# train_ds_list = []
# for degree in ['000', '090', '180', '270']:
#     train_ds_list.append(to_ds_degree(degree))
# train_ds_list.append(to_ds_label('NG'))

LABEL_CLASS_LIST = ['000', '090', '180', '270', 'NG']
label_lookup = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(LABEL_CLASS_LIST, tf.constant([i for i in range(len(LABEL_CLASS_LIST))], dtype=tf.int64)), -1)               
# mirrored_strategy = tf.distribute.MirroredStrategy()
# with mirrored_strategy.scope():
METRICS = [
            tf.keras.metrics.CategoricalAccuracy(name='acc'),
            tf.keras.metrics.Precision(name="precision/000", class_id=0),
            tf.keras.metrics.Precision(name="precision/090", class_id=1),
            tf.keras.metrics.Precision(name="precision/180", class_id=2),
            tf.keras.metrics.Precision(name="precision/270", class_id=3),
            tf.keras.metrics.Precision(name="precision/NG", class_id=4),
            tf.keras.metrics.Recall(name="recall/000", class_id=0),
            tf.keras.metrics.Recall(name="recall/090", class_id=1),
            tf.keras.metrics.Recall(name="recall/180", class_id=2),
            tf.keras.metrics.Recall(name="recall/270", class_id=3),
            tf.keras.metrics.Recall(name="recall/NG", class_id=4),
]
# data_augmentation = tf.keras.Sequential([
#     tf.keras.layers.experimental.preprocessing.RandomZoom(.5, .2),
#     tf.keras.layers.experimental.preprocessing.RandomContrast([1.0 - 0.9, 1.0 + 1.0]),
#     tf.keras.layers.experimental.preprocessing.RandomCrop(192, 192)
# ])

@tf.function
def parse_img(img):
    # img = tf.io.decode_image(img, channels=all_var_dict['target_shape'][-1], 
    #                             dtype=tf.dtypes.float32, expand_animations = False)
    img = tf.io.decode_jpeg(img,channels=1,dct_method='INTEGER_ACCURATE',try_recover_truncated=True)
    img = tf.cast(img, dtype=tf.dtypes.float32) / 255.0
    # img_with_batch = tf.expand_dims(img, axis=0)
    # grad_components = tf.image.sobel_edges(img_with_batch)
    # edg_image = tf.math.reduce_euclidean_norm(grad_components, axis=-1)
    # grad_mag_square = tf.clip_by_value(edg_image, 0., 1.)
    # # grad_mag_components = grad_components**2
    # # grad_mag_square = tf.sqrt(tf.math.reduce_sum(grad_mag_components,axis=-1)) # sum all magnitude components
    # img = tf.squeeze(grad_mag_square, axis=[0])
    img = tf.image.resize_with_pad(img, all_var_dict['target_shape'][1], all_var_dict['target_shape'][0])
    return img

@tf.function
def random_aug_parse_img(x, p=0.5):
    x = tf.io.decode_jpeg(x,channels=1,dct_method='INTEGER_ACCURATE',try_recover_truncated=True)
    x = tf.cast(x, dtype=tf.dtypes.float32) / 255.0
    # img_with_batch = tf.expand_dims(x, axis=0)
    # grad_components = tf.image.sobel_edges(img_with_batch)
    # edg_image = tf.math.reduce_euclidean_norm(grad_components, axis=-1)
    # grad_mag_square = tf.clip_by_value(edg_image, 0., 1.)
    # # grad_mag_components = grad_components**2
    # # grad_mag_square = tf.sqrt(tf.math.reduce_sum(grad_mag_components,axis=-1)) # sum all magnitude components
    # x = tf.squeeze(grad_mag_square, axis=[0]) # this is the image tensor you want
    if tf.random.uniform([]) < p:
        x = tf.image.random_jpeg_quality(x, 0, 100)
        # if tf.random.uniform([]) < p:
        #     x = tf.image.rgb_to_grayscale(x)
        #     x = tf.squeeze(x, axis=-1)
        #     x = tf.stack([x, x, x], axis=-1)
        # if tf.random.uniform([]) < p:
        #     x = tf.image.flip_left_right(x)
        # if tf.random.uniform([]) < p:
        #     x = tf.image.rgb_to_hsv(x)
        # if tf.random.uniform([]) < p:
        #     # x = tf.image.random_saturation(x, 5, 10)
        #     x = tf.image.adjust_saturation(x, random.uniform(0, 1) * 3) # 0-3
        if tf.random.uniform([]) < p:
            x = tf.image.random_brightness(x, 0.5)
            # x = tf.image.adjust_brightness(x, random.uniform(0, 1) / 2) # 0-0.5
        if tf.random.uniform([]) < p:
            x = tf.image.random_contrast(x, 0.1, 2.0)
        # if tf.random.uniform([]) < p:
        #     x = tf.image.random_hue(x, 0.5)
        # if tf.random.uniform([]) < p:
        #     x = tf.image.central_crop(x, central_fraction=(random.uniform(0, 1) + 1 ) / 2) # 0.5-1
    x = tf.image.resize_with_pad(x, all_var_dict['target_shape'][1], all_var_dict['target_shape'][0])
    return x

@tf.function
def label_to_onehot(label):
    label = all_var_dict['ok_lookup'].lookup(label)
    label = tf.one_hot(label, all_var_dict['LABEL_NUM'])
    # label = tf.cast(label, dtype=tf.float32)
    return label

@tf.function
def parse_func(path, label):
    features = {
        'image': parse_img(tf.io.read_file(path)),
        'label': label_to_onehot(label),
    }
    label = features['label']
    return features, label

@tf.function
def parse_func_with_aug(path, label):
    features = {
        'image': random_aug_parse_img(tf.io.read_file(path)),
        'label': label_to_onehot(label),
    }
    label = features['label']
    return features, label

@tf.function
def parse_example(example_proto):    
    image_feature_description = {
        "path": tf.io.FixedLenFeature([], tf.string),
        "label": tf.io.FixedLenFeature([], tf.string),
    }
    features_in_example = tf.io.parse_single_example(example_proto, image_feature_description)
    # features = {
    #     'image': tf.io.read_file(features_in_example["path"]),
    #     'label': label_to_onehot(label),
    # }
    return features_in_example["path"], features_in_example["label"]

dir_basename = 'preprocessed_0121_conv'
all_var_dict = {
    'called_module': 'conv', # trans or conv
    'dir_basename': dir_basename,
    'base_tb_dir': f'/data/aoi-wzs-p1-dip-fa-nvidia/label_heatsink_screw/tb_logs/{dir_basename}/',
    'base_h5_dir': f'/data/aoi-wzs-p1-dip-fa-nvidia/label_heatsink_screw/trained_h5/{dir_basename}/',
    'random_times': 300, 
    'LOG_VERBOSE': False, # only print print places in code
    'RUN_ALL_VERBOSE': False, #
    'CACHE': True, #
    'MP_POLICY': False, #
    'DISTRIBUTED': None, # Not in distributed mode, return specific strategy then means true
    # 'DISTRIBUTED': mirrored_strategy, 
    # Error occurred when finalizing GeneratorDataset iterator: Cancelled: Operation was cancelled
    'EPOCH': 500, #
    'BATCH_SIZE': 64, # Resource exhausted: OOM with batch size:1024, 512
    # 'train_step': 700, 
    'train_total_images': 25000,
    # 'valid_step': 30,
    'shuffle_buffer': 10000, #
    'target_shape': (640, 640, 1), 
    # 'valid_size': 2000, #
    'split_ratio': 0.5,
    'augment': True,
    'data_augmentation': None, # None if not defined
    'METRICS': METRICS, 
    'LABEL_NUM': len(LABEL_CLASS_LIST), # 
    'train_ds_list': [], #
    'val_ds_list': [], #
    'test_ds': [], 
    'gan_ds_list': [], 
    'tfrecords': tfrecords,
    'ok_lookup': label_lookup, #
    # 'hparams_list': HPARAMS_LIST,
    # 'initial_bias': np.log([pos/neg]),
    # 'class_weight': {0: (1 / neg)*(total)/2.0, 1: (1 / pos)*(total)/2.0}
    # 'degree_lookup': degree_lookup, #
    # 'DEGREE_NUM': len(DEGREE_CLASS_LIST),
}

# def test_ds_to_eva(test_ds, batch_size=all_var_dict['BATCH_SIZE']):
#     return test_ds.map(parse_func, tf.data.experimental.AUTOTUNE).batch(batch_size)

def split_to_train_valid(ds, ratio):
    amount = [i for i,_ in enumerate(ds)][-1] + 1
    amount_to_take = int(amount * ratio)
    shuffle_ds = ds.shuffle(amount)
    return shuffle_ds.take(amount_to_take), shuffle_ds.skip(amount_to_take)

def prepare_trainable_ds(train_ds_list=all_var_dict['train_ds_list'],
                            shuffle_buffer=all_var_dict['shuffle_buffer'], 
                            val_ds_list=all_var_dict['val_ds_list'],  
                            split_ratio=all_var_dict['split_ratio'], 
                            tfrecords=all_var_dict['tfrecords'],
                            batch_size=all_var_dict['BATCH_SIZE'], 
                            augment=all_var_dict['augment'], 
                            cache=all_var_dict['CACHE'],
                            train_total_images=all_var_dict['train_total_images']
                        ):
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    if train_ds_list == [] and tfrecords == []:
        return None
    if tfrecords != []:
        for d in map(tf.data.TFRecordDataset, tfrecords):
            train_ds_list.append(d.map(parse_example, AUTOTUNE))
    if val_ds_list == []:
        tar_train_ds_list = []
        for i in range(len(train_ds_list)):
            splitted_take_ds, splitted_skip_ds = split_to_train_valid(train_ds_list[i], split_ratio)
            tar_train_ds_list.append(splitted_take_ds.repeat())
            if i==0:
                valid_ds = splitted_skip_ds
            else:
                valid_ds = valid_ds.concatenate(splitted_skip_ds)
            # print([i for i,_ in enumerate(valid_ds)][-1] + 1)
        balanced_weights = [1/len(tar_train_ds_list) for t in tar_train_ds_list]
        train_ds = tf.data.experimental.sample_from_datasets(tar_train_ds_list, balanced_weights)
        # valid_ds = valid_ds.concatenate(tf.data.TFRecordDataset('/data/aoi-wzs-p1-dip-fa-nvidia/label_heatsink_screw/tfrecord/to_tune/valid.tfrecord').map(parse_example, AUTOTUNE))
    elif len(val_ds_list) == 1:
        tar_train_ds_list = [d.repeat() for d in train_ds_list]
        balanced_weights = [1/len(tar_train_ds_list) for t in tar_train_ds_list]
        train_ds = tf.data.experimental.sample_from_datasets(tar_train_ds_list, balanced_weights)
        valid_ds = val_ds_list[0]
    else:
        tar_train_ds_list = [d.repeat() for d in train_ds_list]
        balanced_weights = [1/len(tar_train_ds_list) for t in tar_train_ds_list]
        train_ds = tf.data.experimental.sample_from_datasets(tar_train_ds_list, balanced_weights)
        for i in range(len(val_ds_list)):
            if i==0:
                valid_ds = val_ds_list[i]
            else:
                valid_ds = valid_ds.concatenate(val_ds_list[i])
    if augment:
        train_ds = train_ds.shuffle(shuffle_buffer).map(parse_func_with_aug, num_parallel_calls=AUTOTUNE)
    else:
        train_ds = train_ds.shuffle(shuffle_buffer).map(parse_func, num_parallel_calls=AUTOTUNE)
    train_ds = train_ds.take(int(train_total_images))
    if cache:
        train_ds = train_ds.cache().batch(batch_size).prefetch(AUTOTUNE)
        valid_ds = valid_ds.map(parse_func, num_parallel_calls=AUTOTUNE).cache().batch(batch_size).prefetch(AUTOTUNE)
    else:
        train_ds = train_ds.batch(batch_size).prefetch(AUTOTUNE)
        valid_ds = valid_ds.map(parse_func, num_parallel_calls=AUTOTUNE).batch(batch_size).prefetch(AUTOTUNE)
    return train_ds, valid_ds

# to tfrecord
def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def parse_metadata_and_serialize_example(path, label):
    image = tf.io.read_file(path)

    features = {
        "path": _bytes_feature([path.numpy()]),
        "image": _bytes_feature([image.numpy()]),
        "label": _bytes_feature([label.numpy()]),
    }
    example_proto = tf.train.Example(
        features=tf.train.Features(feature=features))
    return example_proto.SerializeToString()

def tf_serialize_example(path, label):
    tf_string = tf.py_function(
        parse_metadata_and_serialize_example,
        (path, label),
        tf.string)
    return tf.reshape(tf_string, ())

def generate_training_tfrecord(dataset, export_path):
    dataset = dataset.map(tf_serialize_example, tf.data.experimental.AUTOTUNE)
    writer = tf.data.experimental.TFRecordWriter(export_path)
    writer.write(dataset)

def split_ds_by_ratio_tfrecord(
        train_ds_list=all_var_dict['train_ds_list'],
        split_ratio=all_var_dict['split_ratio'], 
        tfrecords=all_var_dict['tfrecords'],
    ):
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    if train_ds_list == [] and tfrecords == []:
        return None
    if tfrecords != []:
        for d in map(tf.data.TFRecordDataset, tfrecords):
            train_ds_list.append(d.map(parse_example, AUTOTUNE))
    for i in range(len(train_ds_list)):
        splitted_take_ds, splitted_skip_ds = split_to_train_valid(train_ds_list[i], split_ratio)
        if i==0:
            valid_ds = splitted_skip_ds
            train_ds = splitted_take_ds
        else:
            valid_ds = valid_ds.concatenate(splitted_skip_ds)
            train_ds = train_ds.concatenate(splitted_take_ds)
    # valid_ds = valid_ds.concatenate(tf.data.TFRecordDataset('/data/aoi-wzs-p1-dip-fa-nvidia/label_heatsink_screw/tfrecord/to_tune/valid.tfrecord').map(parse_example, AUTOTUNE))
    generate_training_tfrecord(train_ds, '/data/aoi-wzs-p1-dip-fa-nvidia/label_heatsink_screw/tfrecord/to_train/train.tfrecord')
    generate_training_tfrecord(valid_ds, '/data/aoi-wzs-p1-dip-fa-nvidia/label_heatsink_screw/tfrecord/to_train/valid.tfrecord')

# saved tfrecord to train_ds, valid_ds
@tf.function
def parse_example_from_cache_tfrecord(example_proto):    
    image_feature_description = {
        "path": tf.io.FixedLenFeature([], tf.string),
        "image": tf.io.FixedLenFeature([], tf.string),
        "label": tf.io.FixedLenFeature([], tf.string),
    }
    features_in_example = tf.io.parse_single_example(example_proto, image_feature_description)
    features = {
        'image': parse_img(features_in_example['image']),
        'label': label_to_onehot(features_in_example['label']),
    }
    label = features['label']
    return features, label

@tf.function
def aug_parse_example_from_cache_tfrecord(example_proto):    
    image_feature_description = {
        "path": tf.io.FixedLenFeature([], tf.string),
        "image": tf.io.FixedLenFeature([], tf.string),
        "label": tf.io.FixedLenFeature([], tf.string),
    }
    features_in_example = tf.io.parse_single_example(example_proto, image_feature_description)
    features = {
        'image': random_aug_parse_img(features_in_example['image']),
        'label': label_to_onehot(features_in_example['label']),
    }
    label = features['label']
    return features, label

def from_tfrecord_to_train_valid(
        shuffle_buffer=all_var_dict['shuffle_buffer'], 
        batch_size=all_var_dict['BATCH_SIZE'], 
        augment=all_var_dict['augment'], 
        cache=all_var_dict['CACHE'],
        # train_total_images=all_var_dict['train_total_images'],
    ):
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    train_ds = tf.data.TFRecordDataset('/data/aoi-wzs-p1-dip-fa-nvidia/label_heatsink_screw/tfrecord/to_train/train.tfrecord')
    valid_ds = tf.data.TFRecordDataset('/data/aoi-wzs-p1-dip-fa-nvidia/label_heatsink_screw/tfrecord/to_train/valid.tfrecord')
    if augment:
        train_ds = train_ds.shuffle(shuffle_buffer).map(aug_parse_example_from_cache_tfrecord, AUTOTUNE).repeat()
    else:
        train_ds = train_ds.shuffle(shuffle_buffer).map(parse_example_from_cache_tfrecord, AUTOTUNE).repeat()
    # train_ds = train_ds.take(int(train_total_images))
    if cache:
        train_ds = train_ds.cache().batch(batch_size).prefetch(AUTOTUNE)
        valid_ds = valid_ds.shuffle(shuffle_buffer).map(parse_example_from_cache_tfrecord, AUTOTUNE).cache().batch(batch_size).prefetch(AUTOTUNE)
    else:
        train_ds = train_ds.batch(batch_size).prefetch(AUTOTUNE)
        valid_ds = valid_ds.shuffle(shuffle_buffer).map(parse_example_from_cache_tfrecord, AUTOTUNE).batch(batch_size).prefetch(AUTOTUNE)
    return train_ds, valid_ds

all_var_dict.update({
    # 'prepare_function': from_tfrecord_to_train_valid,
    'prepare_function': prepare_trainable_ds,
    # 'test_ds_to_eva': test_ds_to_eva,
})

if __name__ == "__main__":
    # split_ds_by_ratio_tfrecord()
    if all_var_dict['called_module'] == 'conv':
        conv_training.main(all_var_dict)
    elif all_var_dict['called_module'] == 'trans':
        transfer_training.main(all_var_dict)