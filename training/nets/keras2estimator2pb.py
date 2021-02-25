import tensorflow as tf
import os
import argparse
from tensorflow import keras
from keras.models import load_model
from PIL import Image
import numpy as np
import glob
from keras.models import Model
from keras import backend as K

FLAGS = tf.flags.FLAGS
#Environs
MODEL1_H5 = os.environ['MODEL1_H5']
MODEL2_H5 = os.environ['MODEL2_H5']
IMAGE_SIZE = tf.flags.DEFINE_integer("size", 76, "resize image dimension")
BATCH_SIZE = os.environ['BATCH_SIZE']
ESTIMATOR_MODEL_DIR = os.environ['ESTIMATOR_MODEL_DIR']
SAVED_PB_PATH = os.environ['SAVED_PB_PATH']

# (img_width,img_length) = IMAGE_SIZE
img_width = IMAGE_SIZE
img_length = IMAGE_SIZE
def model(hdf5_dir):
    model = load_model(hdf5_dir)
    for l in model.layers:
        l.trainable=False
    return model

# tf.flags.DEFINE_integer("n_class", 2, "total number of output classes")
# tf.flags.DEFINE_integer("batch_size", 256, "batch size")
# tf.flags.DEFINE_integer("prefetch", 1, "prefetch dataset")
# tf.flags.DEFINE_integer("shuffle_buffer", 1000, "dataset shuffle buffer")
# tf.flags.DEFINE_float("lr", 1e-3, "learning rate")
# tf.flags.DEFINE_integer("train_max_steps", 300, "train max steps")
# tf.flags.DEFINE_integer("valid_max_steps", 100, "valid max steps")
# feature_spec = {
#     "image": tf.FixedLenFeature([], tf.string),
#     "label": tf.FixedLenFeature([], tf.int64),
# }
def parse_fn(filepaths,label):
    image_string = tf.read_file(filepaths)
    image_decode = tf.image.decode_bmp(image_string,channels=3)
    image_resized = tf.image.resize_images(image_decode,(img_width,img_length))
    img = tf.image.convert_image_dtype(image_resized, tf.float32)
    img = img / 255.
    features={'image': img }
    return features, label

def input_fn():
    filepaths = sorted(list(glob.glob('/notebooks/notebook/DIP_Data3/AluCapacitance/OK/Angle0-OK/*.bmp')))
    #labels = np.repeat(4, len(filepaths)).reshape(len(filepaths),)
    labels = [[0, 0, 0, 0, 1] for i in filepaths]
    #filepaths = tf.constant(filepaths)
    #labels = tf.convert_to_tensor(np.asarray(labels),tf.uint8)
    dataset = tf.data.Dataset.from_tensor_slices((filepaths,labels))
    dataset = dataset.apply(tf.data.experimental.shuffle_and_repeat(len(filepaths),-1))
    dataset = dataset.apply(tf.data.experimental.map_and_batch(map_func=parse_fn,batch_size=BATCH_SIZE))
    dataset = dataset.prefetch(1)
    return dataset


# def make_input_fn(tfrecord_path):

#     def input_fn():
#         dataset = tf.data.TFRecordDataset(tfrecord_path)
#         dataset = dataset.apply(
#             tf.data.experimental.shuffle_and_repeat(FLAGS.shuffle_buffer, None)
#         )
#         dataset = dataset.apply(
#             tf.data.experimental.map_and_batch(_parse_fn, FLAGS.batch_size,
#             num_parallel_calls=os.cpu_count())
#         )
#         dataset = dataset.prefetch(FLAGS.prefetch)

#         return dataset

#     return input_fn

# model function
def model_fn(features, labels, mode, params):
    images = features["image"]
    model1_output = model(MODEL1_H5)(images)
    pred_class_1 = tf.argmax(model1_output, -1)

    with tf.name_scope("optimizers"):
        optimizer = tf.train.AdamOptimizer(learning_rate=1e-3)
    
    if mode == tf.estimator.ModeKeys.PREDICT:
        mapping_2models = tf.constant(['OK','NG'])
        table = tf.contrib.lookup.index_to_string_table_from_tensor(mapping_2models)
        pred_class_str =  table.lookup(pred_class_1)
        
        predictions = {
            "pred_class": pred_class_str,
            "confidence": tf.reduce_max(model1_output, -1),\
        }
        export_outputs = {"predictions": tf.estimator.export.PredictOutput(predictions)}
        return tf.estimator.EstimatorSpec(mode=mode,\
                                         predictions=predictions,\
                                         export_outputs=export_outputs)
        
    with tf.name_scope("losses"):
        print(labels)
        tf.print(labels)
        #onehot_labels_5 = tf.one_hot(labels,depth=5)
        loss1 = tf.losses.softmax_cross_entropy(onehot_labels=labels,\
                                   logits=model1_output,\
                                   weights=1.0,\
                                   label_smoothing=0,\
                                   scope=None,\
                                   )
    labels_index = tf.argmax(labels, -1)
    if mode == tf.estimator.ModeKeys.TRAIN:
        train_op1 = optimizer.minimize(loss = loss1,global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode = mode,\
                                          loss = loss1,\
                                          train_op = train_op1\
                                         )

    elif mode == tf.estimator.ModeKeys.EVAL:
        metrics = {
            "accuracy": tf.metrics.accuracy(labels_index, pred_class_1),
            "pr-auc": tf.metrics.auc(labels_index, pred_class_1, curve="PR", summation_method="careful_interpolation"),
            "fn": tf.metrics.false_negatives(labels_index, pred_class_1),
            "fp": tf.metrics.false_positives(labels_index, pred_class_1),
            "tn": tf.metrics.true_negatives(labels_index, pred_class_1),
            "tp": tf.metrics.true_positives(labels_index, pred_class_1),
            "precision": tf.metrics.precision(labels_index, pred_class_1),
            "recall": tf.metrics.recall(labels_index, pred_class_1)
        }

        return tf.estimator.EstimatorSpec(mode = mode,
                                          loss = loss1,
                                          eval_metric_ops = metrics)

    
    else:
        return tf.errors.error_code_from_exception_type

# serving input
def serving_input_receiver_fn():
    """An input receiver that expects a serialized tf.Example."""
    receiver_tensors = {
#         "image": tf.placeholder(dtype=tf.float32,shape=[None,76,76,3],name='image_tensors'),\
        "image": tf.placeholder(dtype=tf.string,shape=[None]),\
        #"angle": tf.placeholder(tf.string, (None)),\
    }

    features = {    
        "image": receiver_tensors['image'],\
    }
    
    def serving_parse_fn(image_string):
        image_decode = tf.image.decode_bmp(image_string,channels=3)
        image_resized = tf.image.resize_images(image_decode,[img_length,img_width])
        img = tf.image.convert_image_dtype(image_resized, tf.float32)
        img = img / 255.
        return img
    
    features['image'] = tf.map_fn(serving_parse_fn,features['image'], dtype=tf.float32, back_prop=False)
    return tf.estimator.export.ServingInputReceiver(features, receiver_tensors)




if __name__ == "__main__":
    # training
    estimator = tf.estimator.Estimator(model_fn, ESTIMATOR_MODEL_DIR)

    K.clear_session()
    estimator.train(input_fn, max_steps=1)

    # evaluate
    K.clear_session()
    result = estimator.evaluate(input_fn, steps=1)
    print(result)

    # export saved model
    K.clear_session()
    estimator.export_saved_model(
        SAVED_PB_PATH,
        serving_input_receiver_fn=serving_input_receiver_fn
    )