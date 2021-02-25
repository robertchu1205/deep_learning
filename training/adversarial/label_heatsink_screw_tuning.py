import tensorflow as tf
from tensorboard.plugins.hparams import api as hp
# import neural_structured_learning as nsl
import time, os
import glob
import module.model as model_package

global LABEL_NUM, EPOCH, DIR_BASENAME, BASE_MODEL
LABEL_NUM = int(os.environ['label_num'])
EPOCH = int(os.environ['EPOCH'])
DIR_BASENAME = os.environ['DIR_BASENAME']
BASE_MODEL = os.environ['BASE_MODEL']
IMAGE_WIDTH = int(os.environ['IMAGE_WIDTH'])
COMP = os.environ['COMP']
LEARNING_RATE = float(os.environ['LEARNING_RATE'])
conv_count = int(os.environ['conv_count'])
first_filter = int(os.environ['first_filter']) 
kernel_size = int(os.environ['kernel_size']) 
strides = int(os.environ['strides']) 
maxpool = bool(os.environ['maxpool'])

models_to_be_loaded = {
    'LABEL'    : '/data/aoi-wzs-p1-dip-fa-nvidia/label_heatsink_screw/trained_h5/LABEL/label_M2_18402_5e-7_ep_118-vl_0.0026-va_0.9996.h5', 
    'HEAT_SINK': '/data/aoi-wzs-p1-dip-fa-nvidia/label_heatsink_screw/trained_h5/HEAT_SINK/MobileNet-0.18896-20201127ep_020-vl_0.0-va_1.0.h5', 
    'SCREW'    : '/data/aoi-wzs-p1-dip-fa-nvidia/label_heatsink_screw/trained_h5/SCREW/screw_vgg16_224_ep_200-vl_0.000044-va_1.000000.h5',
    'pre_LABEL': '/data/aoi-wzs-p1-dip-fa-nvidia/label_heatsink_screw/trained_h5/pre_LABEL/cc-3_ff-32_ks-5_s-1_mp-True_asz-0.09968-ep_061-vl_0.0063-va_0.9987.h5',
    # 'pre_LABEL': '/data/aoi-wzs-p1-dip-fa-nvidia/label_heatsink_screw/trained_h5/pre_LABEL/tune_0125_ep_016-vl_0.002942-va_0.999041.h5',
}

# LABEL_CLASS_LIST = ['NG', 'OK']
LABEL_CLASS_LIST = ['000', '090', '180', '270', 'NG']
label_lookup = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(LABEL_CLASS_LIST, tf.constant([i for i in range(len(LABEL_CLASS_LIST))], dtype=tf.int64)), -1)

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
# METRICS = [
#             tf.keras.metrics.CategoricalAccuracy(name='acc'),
#             tf.keras.metrics.Precision(name="precision/ok", class_id=1),
#             tf.keras.metrics.Precision(name="precision/ng", class_id=0),
#             tf.keras.metrics.Recall(name="recall/ok", class_id=1),
#             tf.keras.metrics.Recall(name="recall/ng", class_id=0),
# ]

HP_learning_rate = hp.HParam('learning_rate', hp.RealInterval(1e-5, 5e-5)) 
HP_ReduceLRfactor = hp.HParam('ReduceLRfactor', hp.RealInterval(0.5, 0.8))
HPARAMS_LIST = [HP_learning_rate, HP_ReduceLRfactor]

def parse_img(img):
    # img = tf.io.decode_image(img, channels=all_var_dict['target_shape'][-1], 
    #                             dtype=tf.dtypes.float32, expand_animations = False)
    img = tf.io.decode_jpeg(img,channels=1,dct_method='INTEGER_ACCURATE',try_recover_truncated=True)
    img = tf.cast(img, dtype=tf.dtypes.float32) / 255.0
    img = tf.image.resize_with_pad(img, IMAGE_WIDTH, IMAGE_WIDTH)
    return img

def label_to_onehot(label):
    label = label_lookup.lookup(label)
    label = tf.one_hot(label, LABEL_NUM)
    # label = tf.cast(label, dtype=tf.float32)
    return label

def parse_example(example_proto):    
    image_feature_description = {
        "path": tf.io.FixedLenFeature([], tf.string),
        # "image": tf.io.FixedLenFeature([], tf.string),
        "label": tf.io.FixedLenFeature([], tf.string),
    }
    features_in_example = tf.io.parse_single_example(example_proto, image_feature_description)
    features = {
        'image': parse_img(tf.io.read_file(features_in_example['path'])),
        'label': label_to_onehot(features_in_example['label']),
    }
    label = features['label']
    return features, label

def split_to_train_valid(ds, ratio):
    amount = [i for i,_ in enumerate(ds)][-1] + 1
    amount_to_take = int(amount * ratio)
    shuffle_ds = ds.shuffle(amount)
    return shuffle_ds.take(amount_to_take), shuffle_ds.skip(amount_to_take)

tfrecords = [
    '/data/aoi-wzs-p1-dip-fa-nvidia/label_heatsink_screw/tfrecord/to_tune/000.tfrecord',
    '/data/aoi-wzs-p1-dip-fa-nvidia/label_heatsink_screw/tfrecord/to_tune/090.tfrecord',
    '/data/aoi-wzs-p1-dip-fa-nvidia/label_heatsink_screw/tfrecord/to_tune/180.tfrecord',
    '/data/aoi-wzs-p1-dip-fa-nvidia/label_heatsink_screw/tfrecord/to_tune/270.tfrecord',
    '/data/aoi-wzs-p1-dip-fa-nvidia/label_heatsink_screw/tfrecord/to_tune/NG.tfrecord',
    '/data/aoi-wzs-p1-dip-fa-nvidia/label_heatsink_screw/tfrecord/to_tune/valid.tfrecord',
    '/data/aoi-wzs-p1-dip-fa-nvidia/label_heatsink_screw/tfrecord/to_tune/jan_valid.tfrecord',
]

valid_tfrecords = [
    '/data/aoi-wzs-p1-dip-fa-nvidia/label_heatsink_screw/tfrecord/to_tune/jan.tfrecord',
]

AUTOTUNE = tf.data.experimental.AUTOTUNE

train_ds = []
for i, t in enumerate(tfrecords):
    splitted_take_ds, splitted_skip_ds = split_to_train_valid(tf.data.TFRecordDataset(t), 0.5)
    train_ds.append(splitted_take_ds.repeat())
    if i==0:
        valid_ds = splitted_skip_ds
    else:
        valid_ds = valid_ds.concatenate(splitted_skip_ds)
balanced_weights = [1/len(train_ds) for t in train_ds]
train_ds = tf.data.experimental.sample_from_datasets(train_ds, balanced_weights)
train_ds = train_ds.shuffle(10000).map(parse_example, AUTOTUNE).take(20000).cache().batch(64).prefetch(AUTOTUNE)
if valid_tfrecords != []:
    for vt in valid_tfrecords:
        valid_ds = valid_ds.concatenate(tf.data.TFRecordDataset(vt))
valid_ds = valid_ds.map(parse_example, AUTOTUNE).batch(64).prefetch(AUTOTUNE)

# valid_ds = tf.data.TFRecordDataset('/data/aoi-wzs-p1-dip-fa-nvidia/label_heatsink_screw/tfrecord/to_tune/valid.tfrecord')
# # valid_ds, valid_skip_ds = split_to_train_valid(valid_ds, 0.9)
# valid_ds = valid_ds.map(parse_example, AUTOTUNE).batch(64).prefetch(AUTOTUNE)
# train_ds = [tf.data.TFRecordDataset(t).repeat() for t in tfrecords[:-1]]
# # train_ds.append(valid_skip_ds.repeat())
# train_ds = tf.data.experimental.sample_from_datasets(train_ds,[1/len(train_ds) for i in range(len(train_ds))]).shuffle(10000).map(parse_example, AUTOTUNE)
# train_ds = train_ds.take(20000).cache().batch(64).prefetch(AUTOTUNE)

if __name__ == "__main__":
    # with mirrored_strategy.scope():
    trained_h5_dir = f'''/data/aoi-wzs-p1-dip-fa-nvidia/label_heatsink_screw/trained_h5/{COMP}/{DIR_BASENAME}/'''
    if not os.path.exists(trained_h5_dir):
        os.makedirs(trained_h5_dir)
    for count in range(30):
        hparams = {h: h.domain.sample_uniform() for h in HPARAMS_LIST}
        # init_model = tf.keras.models.load_model(h5)
        # model = model.TransferModel(BASE_MODEL, (IMAGE_WIDTH, IMAGE_WIDTH, 3), LABEL_NUM)
        model = model_package.SomeConv(conv_count, first_filter, kernel_size, strides, 
                                (IMAGE_WIDTH, IMAGE_WIDTH, 1), LABEL_NUM, maxpool)
        model.load_weights(models_to_be_loaded[COMP])
        # adv_config = nsl.configs.make_adv_reg_config(multiplier=2e-1, 
        #                                                 adv_step_size=hparams[HP_adv_step_size], 
        #                                                 adv_grad_norm = 'infinity')
        # model = nsl.keras.AdversarialRegularization(init_model, 
        #                                                 label_keys=['label'], 
        #                                                 adv_config=adv_config)
        optimizer = tf.keras.optimizers.Adam(learning_rate=hparams[HP_learning_rate])
        model.compile(optimizer=optimizer, 
                        loss=tf.keras.losses.CategoricalCrossentropy(),
                        metrics=METRICS)
        lr = hparams[HP_learning_rate] * 1e+7
        to_append_folder_name = (
                                f'rLRf-{str(hparams[HP_ReduceLRfactor])}_' 
                                f'lr-{str(lr)}'
                            )
        tb_log_dir = f'/data/aoi-wzs-p1-dip-fa-nvidia/label_heatsink_screw/tb_logs/{DIR_BASENAME}/{to_append_folder_name}/'
        trained_h5_dir = f'''/data/aoi-wzs-p1-dip-fa-nvidia/label_heatsink_screw/trained_h5/{COMP}/{DIR_BASENAME}/{to_append_folder_name}/'''
        if not os.path.exists(trained_h5_dir):
            os.makedirs(trained_h5_dir)
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss', patience=10, mode='min', 
                restore_best_weights=True,
            ),
            tf.keras.callbacks.TensorBoard(
                log_dir=tb_log_dir,
                write_graph=False,
                profile_batch=0,
            ),
            # hp.KerasCallback(tb_log_dir, hparams),  # log hparams
            tf.keras.callbacks.ModelCheckpoint(
                filepath=trained_h5_dir+'ep_{epoch:03d}-vl_{val_loss:01f}-va_{val_acc:01f}.h5',
                save_best_only=True,
                monitor='val_loss',
                mode='min',
                save_weights_only=True,
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss', factor=hparams[HP_ReduceLRfactor],
                patience=5, min_lr=1e-7,
            ),
        ]
        training_history = model.fit(
                    train_ds, 
                    epochs=EPOCH, 
                    validation_data=valid_ds, 
                    # steps_per_epoch=200, 
                    callbacks=callbacks,
                    verbose=2 # 0 = silent, 1 = progress bar, 2 = one line per epoch
                )
        print('count ', str(count+1), '- hparams - ', hparams, ' :', training_history.history)
    #     tf.keras.models.save_model(model, f'{trained_h5_dir}model', save_format='h5') # save to .pb
        # for d in glob.glob(f'{trained_h5_dir}*.h5'):
        #     model.load_weights(d)
        #     bb = model.base_model
        #     bb.save_weights(d)
        # model.base_model.save_weights(f'{trained_h5_dir}base_model.h5')