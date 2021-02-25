import module.metadata as metadata
import module.conv_training as conv_training
import module.transfer_training as transfer_training
import tensorflow as tf
import sqlite3
import glob, os, sys

METRICS = [
            tf.keras.metrics.CategoricalAccuracy(name='acc'),
            # tf.keras.metrics.Recall(class_id=i, name='recall_'+DEGREE_CLASS_LIST[i]) 
            #                                 for i in range(DEGREE_NUM),
            tf.keras.metrics.Precision(class_id=0, name='precision/OK'),
            tf.keras.metrics.Precision(class_id=1, name='precision/NG-NoneComp'),
            tf.keras.metrics.Precision(class_id=2, name='precision/NG-OutsidePosition'),
            tf.keras.metrics.Precision(class_id=3, name='precision/NG-UpsideDown'),
            tf.keras.metrics.Precision(class_id=4, name='precision/NG-MoreComp'),
            tf.keras.metrics.Recall(class_id=0, name='recall/OK'),
            tf.keras.metrics.Recall(class_id=1, name='recall/NG-NoneComp'),
            tf.keras.metrics.Recall(class_id=2, name='recall/NG-OutsidePosition'),
            tf.keras.metrics.Recall(class_id=3, name='recall/NG-UpsideDown'),
            tf.keras.metrics.Recall(class_id=4, name='recall/NG-MoreComp'),
        ]

LABEL_CLASS_LIST = ['OK', 
                    'NG-NoneComp',
                    'NG-OutsidePosition',
                    'NG-UpsideDown',
                    'NG-MoreComp',
                    ]
LABEL_NUM = len(LABEL_CLASS_LIST)
target_shape = (32, 32, 3)
OK_NG_LOOKUP = tf.constant(list(range(LABEL_NUM)), dtype=tf.int64)
ok_lookup = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(LABEL_CLASS_LIST, OK_NG_LOOKUP), -1)
ng_lookup = ok_lookup
dtyple_tuple = (tf.string, tf.string)

# sqlite3 db
db_path = '/p3/metadata.db'
conn = sqlite3.connect(db_path)
cursor = conn.cursor()
ds_sql_query = {
    'OK':'''
        select path, test_label from metadata
        where test_label = 'OK'
        and component like '%Cap'
        and degree >=0
    ''',
    'NG-NoneComp': '''
        select path, test_label from metadata
        where test_label = 'NG-NoneComp'
        and component like '%Cap'
        and degree >=0
    ''',
    'NG-OutsidePosition': '''
        select path, test_label from metadata
        where test_label = 'NG-OutsidePosition'
        and component like '%Cap'
        and degree >=0
    ''',
    'NG-UpsideDown': '''
        select path, test_label from metadata
        where test_label = 'NG-UpsideDown'
        and component like '%Cap'
        and degree >=0
    ''',
    'NG-MoreComp': '''
        select path, test_label from metadata
        where test_label = 'NG-MoreComp'
        and component like '%Cap'
        and degree >=0
    ''',
    'test': '''
        select path, test_label from metadata
        where test_label != 'NG-InversePolarity' 
        and test_label != 'OK-InvalidPNG' 
        and test_label is not NULL
        and label is NULL
        and component like '%Cap'
        and degree >=0
    ''',
    'other_comp_sql': '''
        select path from metadata
        where (component != 'AluCap' and component != 'ElecCap')
    '''
}

# datasets
ok_ds_list = [tf.data.experimental.SqlDataset('sqlite', db_path, ds_sql_query['OK'], dtyple_tuple)]
ng_ds_list = [tf.data.experimental.SqlDataset('sqlite', db_path, ds_sql_query[lcl], dtyple_tuple) 
                for lcl in LABEL_CLASS_LIST[1:] ]
# other_comp_path_ds = tf.data.experimental.SqlDataset('sqlite', db_path, 
#                 ds_sql_query['other_comp_sql'], (tf.string))
cursor.close()
conn.close()

gan_ds_list = []
# eleccap_list = []
# for p in glob.glob('/p3/unlabeled_elec_m1/*/*.png'):
#     if p.split('/')[-2] != 'NG-InversePolarity':
#         eleccap_list.append(p)
# alucap_list = []
# for p in glob.glob('/p3/unlabeled_alu_m1/*/*.png'):
#     if p.split('/')[-2] != 'NG-InversePolarity':
#         alucap_list.append(p)
# test_ds = tf.data.Dataset.from_tensor_slices(eleccap_list+alucap_list)
# test_ds = []
test_ds = tf.data.experimental.SqlDataset('sqlite', db_path, ds_sql_query['test'], dtyple_tuple)

# directories 
called_module = 'conv'
random_times = 50
if random_times != 0:
    if called_module == 'conv':
        dir_basename = f'cap_m1_conv_r{random_times}'
    elif called_module == 'trans':
        dir_basename = f'cap_m1_trans_r{random_times}'
    else:
        sys.exit('no module matched!')

else:
    if called_module == 'conv':
        dir_basename = f'cap_m1_conv'
    elif called_module == 'trans':
        dir_basename = f'cap_m1_trans'
    else:
        sys.exit('no module matched!')
base_tb_dir = f'/p3/tb_logs/cap-model1/{dir_basename}/'
base_h5_dir = f'/p3/trained_h5/cap-model1/{dir_basename}/'

# all var dict
all_var_dict = {
    'called_module': called_module,
    'dir_basename': dir_basename,
    'base_tb_dir': base_tb_dir,
    'base_h5_dir': base_h5_dir,
    'random_times': random_times, 
    'LOG_VERBOSE': False, # only print print places in code
    'RUN_ALL_VERBOSE': True, 
    'CACHE': True, #
    'MP_POLICY': False, 
    'DISTRIBUTED': True, 
    # Error occurred when finalizing GeneratorDataset iterator: Cancelled: Operation was cancelled
    'METRICS': METRICS, 
    'EPOCH': 100, 
    'BATCH_SIZE': 512, # Resource exhausted: OOM with batch size:1024, 512
    'valid_size': 10000, #
    'train_step': 500, 
    'valid_step': 170, 
    'shuffle_buffer': 10000, #
    'LABEL_NUM': LABEL_NUM, 
    'target_shape': target_shape, 
    'ok_ds': ok_ds_list, #
    'ng_ds': ng_ds_list, #
    'test_ds': test_ds, 
    'gan_ds_list': gan_ds_list, 
    'ok_lookup': ok_lookup, #
    'ng_lookup': ng_lookup, #
}

# alu model 1
def parse_img(img):
    img = tf.io.decode_image(img, channels=all_var_dict['target_shape'][-1], 
                                dtype=tf.dtypes.float32)
    img = tf.image.resize_with_pad(img, all_var_dict['target_shape'][1], 
                                all_var_dict['target_shape'][0])
    # img = tf.cast(img, dtype=tf.float32) / 255.
    return img

def parse_model1(path, label):
    img = tf.io.read_file(path)
    img = parse_img(img)
    if AUGMENT:
        img = metadata.augment_img(img, 6, target_shape)
    label = all_var_dict['ok_lookup'].lookup(label)
    onehot_label = tf.one_hot(label, all_var_dict['LABEL_NUM'], dtype='int64')
    onehot_label = tf.cast(onehot_label, dtype=tf.float32)
    return img, onehot_label

def parse_other_com_to_morecomp(path):
    img = tf.io.read_file(path)
    img = parse_img(img)
    if AUGMENT:
        img = metadata.augment_img(img, 6, target_shape)
    label = all_var_dict['ok_lookup'].lookup(
        tf.constant('NG-MoreComp', dtype=tf.string))
    onehot_label = tf.one_hot(label, all_var_dict['LABEL_NUM'], dtype='int64')
    onehot_label = tf.cast(onehot_label, dtype=tf.float32)
    return img, onehot_label

def prepare_trainable_ds(ok_ds_list=all_var_dict['ok_ds'], 
                            ng_ds_list=all_var_dict['ng_ds'], 
                            shuffle_buffer=all_var_dict['shuffle_buffer'], 
                            valid_size=all_var_dict['valid_size'],  
                            batch_size=all_var_dict['BATCH_SIZE'], 
                            augment=False, cache=all_var_dict['CACHE']):
    global AUGMENT
    AUGMENT = augment
    pro_ok_ds_list = [odl.map(parse_model1, tf.data.experimental.AUTOTUNE)
                        for odl in ok_ds_list]
    pro_ng_ds_list = [ndl.map(parse_model1, tf.data.experimental.AUTOTUNE)
                        for ndl in ng_ds_list]
    # pro_ng_ds_list[3] = tf.data.experimental.sample_from_datasets(
    #     [pro_ng_ds_list[3].repeat(), 
    #     other_comp_path_ds.map(parse_other_com_to_morecomp, tf.data.experimental.AUTOTUNE).repeat()]
    # )
    pro_ds = [d.repeat() for d in (pro_ok_ds_list+pro_ng_ds_list)]
    pro_ds = tf.data.experimental.sample_from_datasets(pro_ds)
    train_ds = pro_ds.skip(valid_size)
    if cache is True:
        train_ds = train_ds.shuffle(shuffle_buffer).cache().batch(
            batch_size).repeat().prefetch(tf.data.experimental.AUTOTUNE)
        valid_ds = pro_ds.take(valid_size).shuffle(shuffle_buffer).cache().batch(
            batch_size).repeat().prefetch(tf.data.experimental.AUTOTUNE)
    else:
        train_ds = train_ds.shuffle(shuffle_buffer).batch(
            batch_size).repeat().prefetch(tf.data.experimental.AUTOTUNE)
        valid_ds = pro_ds.take(valid_size).shuffle(shuffle_buffer).batch(
            batch_size).repeat().prefetch(tf.data.experimental.AUTOTUNE)
    return train_ds, valid_ds

def parse_model1_for_eva(path, label):
    img = tf.io.read_file(path)
    img = parse_img(img)
    label = all_var_dict['ok_lookup'].lookup(label)
    onehot_label = tf.one_hot(label, all_var_dict['LABEL_NUM'], dtype='int64')
    onehot_label = tf.cast(onehot_label, dtype=tf.float32)
    return img, onehot_label

def test_ds_to_eva(test_ds, batch_size=all_var_dict['BATCH_SIZE']):
    test_ds = test_ds.map(parse_model1_for_eva, tf.data.experimental.AUTOTUNE)
    test_ds = test_ds.batch(batch_size)
    return test_ds


# call main in other training
if __name__ == "__main__":
    all_var_dict['prepare_function'] = prepare_trainable_ds
    all_var_dict['test_ds_to_eva'] = test_ds_to_eva
    if all_var_dict['called_module'] == 'conv':
        conv_training.main(all_var_dict)
    elif all_var_dict['called_module'] == 'trans':
        transfer_training.main(all_var_dict)
