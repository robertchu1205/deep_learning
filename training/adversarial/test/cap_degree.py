import module.model as model
import module.metadata as metadata
import module.conv_training as conv_training
import tensorflow as tf
import sqlite3

METRICS = [
            tf.keras.metrics.CategoricalAccuracy(name='acc'),
            # model.CategoricalTruePositives(),
            # tf.keras.metrics.Recall(class_id=i, name='recall_'+DEGREE_CLASS_LIST[i]) 
            #                                 for i in range(DEGREE_NUM),
            tf.keras.metrics.Precision(class_id=0, name='precision/0'),
            tf.keras.metrics.Precision(class_id=1, name='precision/180'),
            tf.keras.metrics.Precision(class_id=2, name='precision/270'),
            tf.keras.metrics.Precision(class_id=3, name='precision/90'),
            tf.keras.metrics.Recall(class_id=0, name='recall/0'),
            tf.keras.metrics.Recall(class_id=1, name='recall/180'),
            tf.keras.metrics.Recall(class_id=2, name='recall/270'),
            tf.keras.metrics.Recall(class_id=3, name='recall/90'),
        ]

# sqlite3 db
db_path = '/p3/metadata.db'
conn = sqlite3.connect(db_path)
cursor = conn.cursor()
degree_sql_query = {
    'OK':'''
        select distinct degree from metadata 
        where label = 'OK'
        and (component = 'AluCap' or component = 'ElecCap')
        and (degree = '0' or degree = '270')
        and width is not NULL
    ''',
    'NG':'''
        select distinct degree from metadata 
        where label = 'NG-InversePolarity'
        and (component = 'AluCap' or component = 'ElecCap')
        and degree >= 0 
    '''
}
size_sql_query = '''
            select distinct width, height from metadata
            where (label = 'OK' or label = 'NG-InversePolarity')
            and (component = 'AluCap' or component = 'ElecCap')
            and (degree = '0' or degree = '270')
            and width is not NULL
        '''
ds_sql_query = {
    'OK':'''
        select path, degree from metadata
        where label = 'OK'
        and (component = 'AluCap' or component = 'ElecCap')
        and (degree = '0' or degree = '270')
        and width is not NULL
    ''',
    'NG': '''
        select path, degree from metadata
        where label = 'NG-InversePolarity'
        and (component = 'AluCap' or component = 'ElecCap')
        and (degree = '0' or degree = '270')
        and width is not NULL
    '''
}

## to avoid datebase locked
DEGREE_CLASS_LIST = metadata.degree_class_list(cursor, degree_sql_query['OK'], degree_sql_query['NG'])
DEGREE_NUM = len(DEGREE_CLASS_LIST)
target_shape = metadata.largest_to_target_shape(cursor, size_sql_query)
ok_lookup, ng_lookup = metadata.lookup_table(DEGREE_CLASS_LIST, [1, 0, 3, 2])
dtyple_tuple = (tf.string, tf.string)

# datasets
ok_ds = tf.data.experimental.SqlDataset('sqlite', db_path, ds_sql_query['OK'], dtyple_tuple)
ng_ds = tf.data.experimental.SqlDataset('sqlite', db_path, ds_sql_query['NG'], dtyple_tuple)
# test_ds = tf.data.experimental.SqlDataset('sqlite', db_path, ds_sql_query['TEST'], dtype_tuple)
cursor.close()
conn.close()
gan_tfrecord_list = [
    '/p3/tfrecord/alu-m2/Alu-M2-08231119-Stylized.tfrecord', 
    '/p3/tfrecord/alu-m2/dcgan_alu_m2_ng_img.tfrecord'
]
shuffle_buffer=10000
gan_ds_list = metadata.to_ds_list(gan_tfrecord_list, shuffle_buffer, target_shape)

# directories 
random_times = 3
if random_times != 0:
    dir_basename = f'cap_deg_r{random_times}'
else:
    dir_basename = f'cap_deg'
base_tb_dir = f'/p3/tb_logs/cap-degree/{dir_basename}/'
base_h5_dir = f'/p3/trained_h5/cap-degree/{dir_basename}/'

# all var dict
all_var_dict = {
    'dir_basename': dir_basename,
    'base_tb_dir': base_tb_dir,
    'base_h5_dir': base_h5_dir,
    'random_times': random_times, 
    'LOG_VERBOSE': False, # only print print places in code
    'RUN_ALL_VERBOSE': False, 
    'MP_POLICY': False, 
    'DISTRIBUTED': False, 
    # Error occurred when finalizing GeneratorDataset iterator: Cancelled: Operation was cancelled
    'METRICS': METRICS, 
    'EPOCH': 100, 
    'BATCH_SIZE': 64, # Resource exhausted: OOM with batch size:1024, 512
    'valid_size': 20000, 
    'train_step': 1600, 
    'valid_step': 500, 
    'shuffle_buffer': shuffle_buffer, 
    'LABEL_NUM': DEGREE_NUM, 
    'target_shape': target_shape, 
    'ok_ds': ok_ds, 
    'ng_ds': ng_ds, 
    # 'test_ds': test_ds, 
    'gan_ds_list': gan_ds_list, 
    'ok_lookup': ok_lookup, 
    'ng_lookup': ng_lookup, 
}

# cap degree class4
def process_ok(path, degree):
    byte_string_img = tf.io.read_file(path)
    img = tf.io.decode_image(byte_string_img, channels=target_shape[-1], dtype=tf.dtypes.float32)
#     img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize_with_crop_or_pad(img, target_shape[1], target_shape[0])
    if AUGMENT:
        img = metadata.augment_img(img, 6, target_shape)
    deg = ok_lookup.lookup(degree)
    onehot_degree = tf.one_hot(deg, DEGREE_NUM, dtype='int64')
    return img, onehot_degree

def process_ng(path, degree):
    byte_string_img = tf.io.read_file(path)
    img = tf.io.decode_image(byte_string_img, channels=target_shape[-1], dtype=tf.dtypes.float32)
#     img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize_with_crop_or_pad(img, target_shape[1], target_shape[0])
    if AUGMENT:
        img = metadata.augment_img(img, 6, target_shape)
    deg = ng_lookup.lookup(degree)
    onehot_degree = tf.one_hot(deg, DEGREE_NUM, dtype='int64')
    return img, onehot_degree

def prepare_trainable_ds(ok_ds, ng_ds, gan_ratio, gan_ds_list, 
                            gan_each_ratio_list, shuffle_buffer, valid_size,  
                            batch_size, augment=False, cache=True):
    global AUGMENT
    AUGMENT = augment
    pro_ok_ds = ok_ds.shuffle(shuffle_buffer).map(process_ok, 
                                                    tf.data.experimental.AUTOTUNE)
    pro_ng_ds = ng_ds.shuffle(shuffle_buffer).map(process_ng, 
                                                    tf.data.experimental.AUTOTUNE)
    pro_ds = tf.data.experimental.sample_from_datasets(
            [pro_ok_ds.repeat(), pro_ng_ds.repeat()], weights=[0.5, 0.5])
    if gan_ratio == 0.:
        train_ds = pro_ds.skip(valid_size)
    else:
        combined_gan_ds = tf.data.experimental.sample_from_datasets(
                                gan_ds_list, weights=gan_each_ratio_list)
        filtered_ng_ds = pro_ds.skip(valid_size).filter(lambda image, degree: 
                    tf.math.argmax(input = degree)==1 or tf.math.argmax(input = degree)==3)
        filtered_ok_ds = pro_ds.skip(valid_size).filter(lambda image, degree: 
                    tf.math.argmax(input = degree)==0 or tf.math.argmax(input = degree)==2)
        mixed_ng_ds = tf.data.experimental.sample_from_datasets(
            [filtered_ng_ds.repeat(), combined_gan_ds.repeat()], 
            weights=[1 - gan_ratio, gan_ratio])
        train_ds = tf.data.experimental.sample_from_datasets(
            [filtered_ok_ds.repeat(), mixed_ng_ds.repeat()], weights=[0.5, 0.5])
    if cache is True:
        train_ds = train_ds.cache().repeat().batch(batch_size).prefetch(
            buffer_size=tf.data.experimental.AUTOTUNE)
        valid_ds = pro_ds.take(valid_size).cache().repeat().batch(batch_size).prefetch(
            buffer_size=tf.data.experimental.AUTOTUNE)
    else:
        train_ds = train_ds.repeat().batch(batch_size).prefetch(
            buffer_size=tf.data.experimental.AUTOTUNE)
        valid_ds = pro_ds.take(valid_size).repeat().batch(batch_size).prefetch(
            buffer_size=tf.data.experimental.AUTOTUNE)
    return train_ds, valid_ds

# train_valid_data = tfrecord.mix_train_dataset_with_generative_from_tfrecord(
#         ori_img_tfrecord_list, stylized_img_tfrecord_list, 
#         dcgan_img_tfrecord_list, valid_img_tfrecord_list,
#         all_var_dict['target_shape'], hparams[HP_dcgan_ratio], 
#         hparams[HP_generative_ratio], all_var_dict['BATCH_SIZE'], hparams[HP_RESIZE_METHOD], 
#         cache=True, shuffle_buffer_size=all_var_dict['shuffle_buffer'])
# test_data = tfrecord.test_dataset_from_tfrecord(test_img_tfrecord_list, all_var_dict['target_shape'],
#         all_var_dict['BATCH_SIZE'], hparams[HP_RESIZE_METHOD], 
#         cache=True, shuffle_buffer_size=all_var_dict['shuffle_buffer'])

# call main in other training
if __name__ == "__main__":
    all_var_dict['prepare_function'] = prepare_trainable_ds
    # all_var_dict['test_ds_to_eva'] = test_ds_to_eva
    conv_training.main(all_var_dict)