from module.metadata import augment_img
import module.conv_training as conv_training
import module.transfer_training as transfer_training
import tensorflow as tf
import sqlite3
import glob, os, sys

# tf2.2
mirrored_strategy = tf.distribute.MirroredStrategy(
    cross_device_ops=tf.distribute.HierarchicalCopyAllReduce())
multiworker_strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy(
    tf.distribute.experimental.CollectiveCommunication.NCCL)
# mirrored_strategy = tf.distribute.MirroredStrategy(
#     devices=['/gpu:0', '/gpu:1']
# )

# METRICS = [
#             tf.keras.metrics.CategoricalAccuracy(name='acc'),
#             # tf.keras.metrics.Recall(class_id=i, name='recall_'+DEGREE_CLASS_LIST[i]) 
#             #                                 for i in range(DEGREE_NUM),
#             tf.keras.metrics.Precision(class_id=0, name='precision/OK'),
#             tf.keras.metrics.Precision(class_id=1, name='precision/NG-NoneComp'),
#             tf.keras.metrics.Precision(class_id=2, name='precision/NG-OutsidePosition'),
#             tf.keras.metrics.Precision(class_id=3, name='precision/NG-UpsideDown'),
#             tf.keras.metrics.Precision(class_id=4, name='precision/NG-MoreComp'),
#             tf.keras.metrics.Precision(class_id=5, name='precision/NG-InversePolarity'),
#             tf.keras.metrics.Recall(class_id=0, name='recall/OK'),
#             tf.keras.metrics.Recall(class_id=1, name='recall/NG-NoneComp'),
#             tf.keras.metrics.Recall(class_id=2, name='recall/NG-OutsidePosition'),
#             tf.keras.metrics.Recall(class_id=3, name='recall/NG-UpsideDown'),
#             tf.keras.metrics.Recall(class_id=4, name='recall/NG-MoreComp'),
#             tf.keras.metrics.Recall(class_id=5, name='recall/NG-InversePolarity'),
#         ]

with mirrored_strategy.scope():
    METRICS = [
                tf.keras.metrics.BinaryAccuracy(name='acc'),
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall'),
                tf.keras.metrics.AUC(name='auc'),
            ]

target_shape = (96, 96, 3)
LABEL_CLASS_LIST = ['OK', 
                    'Overkill',
                    'NG-NoneComp',
                    'NG-OutsidePosition',
                    'NG-UpsideDown',
                    'NG-MoreComp',
                    'NG-InversePolarity',
                    'NG',
                    'Leak',
]

LABEL_TABLE = tf.constant([1, 1, 0, 0, 0, 0, 0, 0, 0], dtype=tf.int64)
ok_lookup = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(LABEL_CLASS_LIST, LABEL_TABLE), -1)
ng_lookup = ok_lookup
DEGREE_CLASS_LIST = ['0', '180', '270', '90']
DEGREE_TABLE = tf.constant(list(range(len(DEGREE_CLASS_LIST))), dtype=tf.int64)
degree_lookup = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(DEGREE_CLASS_LIST, DEGREE_TABLE), -1)
dtyple_tuple = (tf.string, tf.string, tf.string)

# sqlite3 db
db_path = '/tf/robertnb/training/metadata.db'
train_ds_list = []
# for label in LABEL_CLASS_LIST:
#     train_ds_list.append(tf.data.experimental.SqlDataset(
#         'sqlite', db_path,
#         f'''select path, degree, test_label from metadata
#         where
#             degree >= 0 and
#             test_label = '{label}' and
#             component like '%Cap' and
#             extension = 'png' and
#             label is not NULL and 
#             test_label is not NULL
#         ''', dtyple_tuple))

#ok
train_ds_list.append(tf.data.experimental.SqlDataset(
        'sqlite', db_path,
        f'''select path, degree, test_label from metadata
        where
            degree >= 0 and
            test_label = 'OK' and
            component like '%Cap' and
            extension = 'png'
        ''', dtyple_tuple))

#ng
train_ds_list.append(tf.data.experimental.SqlDataset(
        'sqlite', db_path,
        f'''select path, degree, test_label from metadata
        where
            degree >= 0 and
            test_label != 'OK' and
            test_label != 'OK-InvalidPNG' and 
            test_label is not NULL and
            component like '%Cap' and
            extension = 'png'
        ''', dtyple_tuple))
    
test_sql_query =  '''
        select path, degree, A4_result from new_metadata 
        where 
            component like '%Cap' and
            extension = 'png' and 
            degree >= 0 
'''
test_ds = tf.data.experimental.SqlDataset('sqlite', db_path, test_sql_query, dtyple_tuple)

# directories 
basename_prefig = 'AddandConcat'
called_module = 'trans'
random_times = 20
if random_times != 0:
    if called_module == 'conv':
        dir_basename = f'{basename_prefig}_conv_r{random_times}'
    elif called_module == 'trans':
        dir_basename = f'{basename_prefig}_trans_r{random_times}'
    else:
        sys.exit('no module matched!')
else:
    if called_module == 'conv':
        dir_basename = f'{basename_prefig}_cap_all_conv'
    elif called_module == 'trans':
        dir_basename = f'{basename_prefig}_cap_all_trans'
    else:
        sys.exit('no module matched!')

base_tb_dir = f'/p3/tb_logs/2class-cap-all/{dir_basename}/'
base_h5_dir = f'/p3/trained_h5/2class-cap-all/{dir_basename}/'

# all var dict
all_var_dict = {
    'called_module': called_module,
    'dir_basename': dir_basename,
    'base_tb_dir': base_tb_dir,
    'base_h5_dir': base_h5_dir,
    'random_times': random_times, 
    'LOG_VERBOSE': False, # only print print places in code
    'RUN_ALL_VERBOSE': False, #
    'CACHE': True, #
    'MP_POLICY': False, #
    # DISTRIBUTED None: Not in distributed mode, return specific strategy then means true
    'DISTRIBUTED': mirrored_strategy, 
    # Error occurred when finalizing GeneratorDataset iterator: Cancelled: Operation was cancelled
    'METRICS': METRICS, 
    'EPOCH': 100, #
    'BATCH_SIZE': 32, # Resource exhausted: OOM with batch size:1024, 512
    'valid_size': 20000, #
    'train_step': 800, 
    # 'valid_step': 170, 
    'shuffle_buffer': 2000, #
    'LABEL_NUM': 2, # 
    'target_shape': target_shape, 
    'train_ds_list': train_ds_list, #
    'val_ds_list': [test_ds], #
    'test_ds': [], 
    'gan_ds_list': [], 
    'ok_lookup': ok_lookup, #
    'ng_lookup': ng_lookup, #
    'degree_lookup': degree_lookup, #
    'DEGREE_NUM': len(DEGREE_CLASS_LIST),
}

def parse_img_with_aug(img):
    img = tf.io.decode_image(img, channels=all_var_dict['target_shape'][-1], 
                                dtype=tf.dtypes.float32, expand_animations = False)
    img = augment_img(img, 16, 200, all_var_dict['target_shape'])
    # img = tf.image.resize_with_pad(img, all_var_dict['target_shape'][1], 
    #                             all_var_dict['target_shape'][0])
    return img

def parse_img(img):
    img = tf.io.decode_image(img, channels=all_var_dict['target_shape'][-1], 
                                dtype=tf.dtypes.float32, expand_animations = False)
    img = tf.image.resize(img, (all_var_dict['target_shape'][1], all_var_dict['target_shape'][0]), 
                                method=tf.image.ResizeMethod.BICUBIC)
    # img = tf.image.resize_with_pad(img, 
    #                             all_var_dict['target_shape'][1], 
    #                             all_var_dict['target_shape'][0])
    return img

def label_to_onehot(label):
    label = all_var_dict['ok_lookup'].lookup(label)
    onehot_label = tf.one_hot(label, all_var_dict['LABEL_NUM'], dtype='int64')
    onehot_label = tf.cast(onehot_label, dtype=tf.float32)
    return onehot_label

def degree_to_onehot(degree):
    degree = tf.strings.as_string(
        tf.math.abs(tf.strings.to_number(degree, out_type=tf.dtypes.int64)))
    degree = all_var_dict['degree_lookup'].lookup(degree)
    onehot_degree = tf.one_hot(degree, all_var_dict['DEGREE_NUM'], dtype='int64')
    onehot_degree = tf.cast(onehot_degree, dtype=tf.float32)
    return onehot_degree

def train_parse_func(path, degree, label):
    features = {
        'image': parse_img_with_aug(tf.io.read_file(path)),
        'degree': degree_to_onehot(degree),
        'label': label_to_onehot(label),
    }
    return features, features['label']

def test_parse_func(path, degree, label):
    features = {
        'image': parse_img(tf.io.read_file(path)),
        'degree': degree_to_onehot(degree),
        'label': label_to_onehot(label),
    }
    return features, features['label']

def prepare_trainable_ds(train_ds_list=all_var_dict['train_ds_list'], 
                            val_ds_list=all_var_dict['val_ds_list'], 
                            shuffle_buffer=all_var_dict['shuffle_buffer'], 
                            valid_size=all_var_dict['valid_size'],  
                            batch_size=all_var_dict['BATCH_SIZE'], 
                            augment=False, cache=all_var_dict['CACHE']):
    train_pre_ds = [d.map(train_parse_func, tf.data.experimental.AUTOTUNE).repeat() for d in train_ds_list]
    train_pre_ds = tf.data.experimental.sample_from_datasets(train_pre_ds).shuffle(shuffle_buffer)
    if val_ds_list == []:
        train_ds = train_pre_ds.skip(valid_size)
        valid_ds = train_pre_ds.take(valid_size)
    elif len(val_ds_list) == 1:
        train_ds = train_pre_ds
        valid_ds = val_ds_list[0].map(test_parse_func, tf.data.experimental.AUTOTUNE)
    else:
        val_pre_ds = [d.map(train_parse_func, tf.data.experimental.AUTOTUNE).repeat() for d in val_ds_list]
        val_pre_ds = tf.data.experimental.sample_from_datasets(val_pre_ds).shuffle(shuffle_buffer)
        train_ds = train_pre_ds
        valid_ds = val_pre_ds.take(valid_size)
    if cache is True:
        train_ds = train_ds.cache()
        valid_ds = valid_ds.cache()
    train_ds = train_ds.batch(
        batch_size).prefetch(tf.data.experimental.AUTOTUNE)
    valid_ds = valid_ds.batch(
        batch_size).prefetch(tf.data.experimental.AUTOTUNE)
    return train_ds, valid_ds

def test_ds_to_eva(test_ds, batch_size=all_var_dict['BATCH_SIZE']):
    return test_ds.map(test_parse_func, tf.data.experimental.AUTOTUNE).batch(batch_size)

# call main in other training
if __name__ == "__main__":
    all_var_dict['prepare_function'] = prepare_trainable_ds
    all_var_dict['test_ds_to_eva'] = test_ds_to_eva
    if all_var_dict['called_module'] == 'conv':
        conv_training.main(all_var_dict)
    elif all_var_dict['called_module'] == 'trans':
        transfer_training.main(all_var_dict)