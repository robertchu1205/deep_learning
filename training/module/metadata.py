import tensorflow as tf
import sqlite3
import module.tfrecord as tfrecord
import random

# def assign_global(all_var_list):
#     global LABEL_NUM
#     global ok_lookup, ng_lookup
#     global target_shape
#     LABEL_NUM = all_var_list['LABEL_NUM']
#     ok_lookup = all_var_list['ok_lookup']
#     ng_lookup = all_var_list['ng_lookup']
#     target_shape = all_var_list['target_shape']

def degree_class_list(c, ok_sql_query, ng_sql_query):
    LABEL_CLASS_LIST = []
    ok_fetch = c.execute(ok_sql_query).fetchall()
    ng_fetch = c.execute(ng_sql_query).fetchall()
    for i, in ok_fetch:
        LABEL_CLASS_LIST.append(f'{i}')
    for i, in ng_fetch:
        d = str((int(i) + 180) % 360)
        LABEL_CLASS_LIST.append(f'{d}')
    LABEL_NUM = len(LABEL_CLASS_LIST)
    return sorted(LABEL_CLASS_LIST)

def lookup_table(LABEL_CLASS_LIST, ng_index_list):
    OK_DEGREE_INDEX_LOOKUP = tf.constant(list(range(len(LABEL_CLASS_LIST))), dtype=tf.int64)
    NG_DEGREE_INDEX_LOOKUP = tf.constant(ng_index_list, dtype=tf.int64)
    ok_lookup = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(LABEL_CLASS_LIST, OK_DEGREE_INDEX_LOOKUP), -1)
    ng_lookup = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(LABEL_CLASS_LIST, NG_DEGREE_INDEX_LOOKUP), -1)
    return ok_lookup, ng_lookup

def largest_to_target_shape(c, size_sql_query):
    all_wh_list = c.execute(size_sql_query).fetchall()
    biggest_size = 0
    for w, h in all_wh_list:
        if int(h) > biggest_size:
            biggest_size = int(h)
        if int(w) > biggest_size:
            biggest_size = int(w)
    target_shape = (biggest_size, biggest_size, 3)
    return (biggest_size, biggest_size, 3)

def to_ds_list(tfrecord_list, shuffle_buffer, target_shape):
    ds_list = []
    for tfr in tfrecord_list:
        ds = tf.data.TFRecordDataset(tfr).shuffle(shuffle_buffer)
        ds_list.append(ds.map(lambda x: tfrecord.parse_tfrecord_dataset(x, target_shape), 
              num_parallel_calls=tf.data.experimental.AUTOTUNE))
    ds_list = [d.repeat() for d in ds_list]
    return ds_list

def prepare_test_ds(db_path, ds, shuffle_buffer, batch_size, cache=True):
    pro_ds = ok_ds.shuffle(shuffle_buffer).map(process_ok, 
                                                    tf.data.experimental.AUTOTUNE)
    if cache is True:
        test_ds = pro_ds.cache().repeat().batch(batch_size).prefetch(
            buffer_size=tf.data.experimental.AUTOTUNE)
    else:
        test_ds = pro_ds.repeat().batch(batch_size).prefetch(
            buffer_size=tf.data.experimental.AUTOTUNE)
    return test_ds

# random_color, image_strengthen, blur, deformation
def augment_img(img, pad, ratio, target_shape):
    # Add {pad} pixels of padding
    # img = tf.image.resize_with_crop_or_pad(img, target_shape[1]+pad, target_shape[0]+pad)
    img = tf.image.resize(img, (target_shape[1]+pad, target_shape[0]+pad), 
                                method=tf.image.ResizeMethod.BICUBIC) 
    # Random crop back to original target shape
    img = tf.image.random_crop(img, size=[target_shape[1], target_shape[0], target_shape[-1]])
    odd = random.randint(1, ratio) # 0.1 become grayscale
    if odd == 1:
        # img = tf.image.rgb_to_grayscale(img)
        img = img[:,:,0]
        img = tf.stack([img, img, img], axis=-1)
    # Random brightness
    # img = tf.image.random_brightness(img, max_delta=0.5) 
    # Random saturation
    # img = tf.image.random_saturation(img, 0., 3.)
    return img