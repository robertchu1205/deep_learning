from __future__ import absolute_import, division, print_function, unicode_literals
# import IPython.display as display
import tensorflow as tf
# from PIL import Image
import numpy as np
# import matplotlib.pyplot as plt
import os
import glob
import time
import datetime
import model
import logging
# import servable_model
import argparse
import json

AUTOTUNE = tf.data.experimental.AUTOTUNE

###

def create_labels(train_root):
    CLASS_NAMES = sorted(item.split('/')[-1] for item in glob.glob(train_root+'/*') if os.path.isdir(item))
    CLASS_NAMES_INDEX = tf.constant(
        list(range(len(CLASS_NAMES))), dtype=tf.int64)
    label_to_index = dict((name, index) for index, name in enumerate(CLASS_NAMES))
    return (CLASS_NAMES, CLASS_NAMES_INDEX, label_to_index)

def get_label(file_path):
    # convert the path to a list of path components
    parts = tf.strings.split(file_path, '/')
    # The second to last is the class-directory
    return parts[-2] == CLASS_NAMES

def decode_img(img):
    # convert the compressed string to a 3D uint8 tensor
    img = tf.io.decode_png(img, channels=int(os.environ['CHANNELS']))
    # img = tf.io.decode_image(img, channels=int(os.environ['CHANNELS']), expand_animations=False)
    # Use `convert_image_dtype` to convert to floats in the [0,1] range.
    img = tf.image.convert_image_dtype(img, tf.float32)
    # resize the image to the desired size.
    img = tf.image.resize(img, [int(os.environ['IMG_WIDTH']), int(os.environ['IMG_HEIGHT'])])
    return img

def filepathTOb64(img):
    # img = tf.io.encode_base64(img)
    # img = tf.compat.as_str_any(img)
    # with open(filename, "rb") as f:
    #     image = f.read()
    #     image = base64.b64encode(image).decode("utf-8") # base64 string
    #     image = image.encode()
    #     image = base64.decodebytes(image)
    #     image = base64.urlsafe_b64encode(image)
    return img

def process_path(file_path):
    label = get_label(file_path)
    # load the raw data from the file as a string
    img = tf.io.read_file(file_path)
    img = decode_img(img) # input tensor is numpy array
    # img = filepathTOb64(img) # input tensor is b64 string
    return img, label

# For well shuffled, batched which gonna to be available
def prepare_for_training(ds, cache=True, shuffle_buffer_size=200000):
    # This is a small dataset, only load it once, and keep it in memory.
    # use `.cache(filename)` to cache preprocessing work for datasets that don't
    # fit in memory.
    if cache:
        if isinstance(cache, str):
            ds = ds.cache(cache)
        else:
            ds = ds.cache()
    ds = ds.shuffle(buffer_size=shuffle_buffer_size)
    # Repeat forever
    ds = ds.repeat()
    ds = ds.batch(int(os.environ['BATCH_SIZE']))
    # `prefetch` lets the dataset fetch batches in the background while the model
    # is training.
    ds = ds.prefetch(buffer_size=AUTOTUNE)
    return ds

def plot_history(history):
    fig, axes = plt.subplots(2, 2, figsize=(10, 6))

    axes[0, 0].set_title("loss")
    axes[0, 0].plot(history["loss"])
    axes[0, 0].plot(history["val_loss"])
    axes[0, 0].legend(["train", "val"])

    axes[0, 1].set_title("accuracy")
    axes[0, 1].plot(history["accuracy"])
    axes[0, 1].plot(history["val_accuracy"])
    axes[0, 1].legend(["train", "val"])

    axes[1, 0].set_title("recall")
    axes[1, 0].plot(history["recall_score"])
    axes[1, 0].plot(history["val_recall_score"])
    axes[1, 0].legend(["train", "val"])

    axes[1, 1].set_title("precision")
    axes[1, 1].plot(history["precision_score"])
    axes[1, 1].plot(history["val_precision_score"])
    axes[1, 1].legend(["train", "val"])

    fig.show()

def to_its_ds(root, cache=True):
    root_dirs = glob.glob(root+'/*/*')
    list_ds = tf.data.Dataset.list_files(root+'/*/*')
    # Set `num_parallel_calls` so multiple images are loaded/processed in parallel.
    labeled_ds = list_ds.map(process_path, num_parallel_calls=AUTOTUNE)
    image_label_ds = prepare_for_training(labeled_ds, cache=cache, shuffle_buffer_size=len(root_dirs))
    return image_label_ds

def split_ds(tobesplit_root, val_percent, test_percent):
    list_ds = tf.data.Dataset.list_files(tobesplit_root+'/*/*')
    unhandled_labled_ds = list_ds.map(process_path, num_parallel_calls=AUTOTUNE)
    train_percent = 1 - val_percent - test_percent
    total_batch_count = np.ceil(len(glob.glob(tobesplit_root+'/*/*'))/int(os.environ['BATCH_SIZE']))
    os.environ['VALIDATION_STEPS'] = str(int(np.ceil(total_batch_count * val_percent))) # val_batch_count
    os.environ['TEST_STEPS'] = str(int(np.ceil(total_batch_count * test_percent))) # test_batch_count
    os.environ['STEPS_PER_EPOCH'] = str(int(total_batch_count - int(os.environ['VALIDATION_STEPS']) - int(os.environ['TEST_STEPS']))) # train_batch_count
    val_ds = prepare_for_training(unhandled_labled_ds.take(int(os.environ['VALIDATION_STEPS'])), cache=True, shuffle_buffer_size=int(np.ceil(len(glob.glob(tobesplit_root+'/*/*')) * val_percent)))
    test_ds = prepare_for_training(unhandled_labled_ds.skip(int(os.environ['VALIDATION_STEPS'])).take(int(os.environ['TEST_STEPS'])), cache=True, shuffle_buffer_size=int(np.ceil(len(glob.glob(tobesplit_root+'/*/*')) * test_percent)))
    train_ds = prepare_for_training(unhandled_labled_ds.skip(int(os.environ['VALIDATION_STEPS'])).skip(int(os.environ['TEST_STEPS'])), cache=True, shuffle_buffer_size=int(np.ceil(len(glob.glob(tobesplit_root+'/*/*')) * train_percent)))
    total_ds = prepare_for_training(unhandled_labled_ds, cache=True, shuffle_buffer_size=int(np.ceil(len(glob.glob(tobesplit_root+'/*/*')))))
    # train_ds = unhandled_labled_ds.skip(int(os.environ['VALIDATION_STEPS'])).skip(int(os.environ['TEST_STEPS'])).cache()
    return (total_ds, train_ds, val_ds, test_ds)

def configjson_to_environ():
    try:
        with open('/config/config.json', 'r') as f: #For Docker build
            config = json.load(f)
    except Exception as e:
        logging.error("config.json doesn't exist. Exception says:' {e} '".format(e=e))
    os.environ['TRAIN_ROOT'] = config['TRAIN_ROOT']
    os.environ['VAL_ROOT'] = config['VAL_ROOT']
    os.environ['TEST_ROOT'] = config['TEST_ROOT']
    os.environ['MODEL_DIR'] = config['MODEL_DIR']
    os.environ['BATCH_SIZE'] = str(config['BATCH_SIZE'])
    os.environ['EPOCH'] = str(config['EPOCH'])
    os.environ['CHANNELS'] = str(config['CHANNELS'])
    # os.environ['IMG_WIDTH'] = str(config['IMG_WIDTH'])
    # os.environ['IMG_HEIGHT'] = str(config['IMG_HEIGHT'])
    os.environ['VAL_PERCENT'] = str(config['VAL_PERCENT'])
    os.environ['TEST_PERCENT'] = str(config['TEST_PERCENT'])
    # os.environ['maxbool'] = str(config['maxbool'])
    # os.environ['layers'] = str(config['layers'])
    # os.environ['filters'] = str(config['filters'])
    if config['split'] and \
        os.environ['TRAIN_ROOT']!="" and \
        config["VAL_ROOT"]=="" and \
        config["TEST_ROOT"]=="" and \
        (config["VAL_PERCENT"] > 0 and config["VAL_PERCENT"] < 1) and \
        (config["TEST_PERCENT"] > 0 and config["TEST_PERCENT"] < 1):
        return (True, int(os.environ['layers']), int(os.environ['filters']), bool(os.environ['maxbool']))
    elif not config['split'] and \
        os.environ['TRAIN_ROOT']!="" and config["VAL_ROOT"]!="" and config["TEST_ROOT"]!="":
        return (False, int(os.environ['layers']), int(os.environ['filters']), bool(os.environ['maxbool']))
    else:
        print("Please Correct Root in config.json")
        exit()


if __name__ == "__main__":
    try:
        (split_ornot, layers, filters, maxbool) = configjson_to_environ()
        if not os.path.exists('/'+os.environ['MODEL_DIR']):
            os.makedirs('/'+os.environ['MODEL_DIR'])
        logging.basicConfig(filename='/{td}/model_log.log'.format(td=os.environ['MODEL_DIR']),
                                filemode='a',
                                format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                                datefmt='%Y-%m-%d %H:%M:%S',
                                level=logging.DEBUG)
    except Exception as e:
        print("config.json doesn't exist. Exception says:' {e} '".format(e=e))
        exit()
    (CLASS_NAMES, CLASS_NAMES_INDEX, label_to_index) = create_labels(os.environ['TRAIN_ROOT'])
    if split_ornot is True:
        # Split train dataset by ratio
        (total_ds, train_ds, val_ds, test_ds) = split_ds(os.environ['TRAIN_ROOT'], float(os.environ['VAL_PERCENT']), float(os.environ['TEST_PERCENT']))
    else:
        # Divided before loading dataset
        train_ds = to_its_ds(os.environ['TRAIN_ROOT'])
        val_ds = to_its_ds(os.environ['VAL_ROOT'])
        test_ds = to_its_ds(os.environ['TEST_ROOT'])
        os.environ['STEPS_PER_EPOCH'] = str(np.ceil(len(glob.glob(train_root+'/*/*'))/int(os.environ['BATCH_SIZE']))) # top
        os.environ['VALIDATION_STEPS'] = str(np.ceil(len(glob.glob(val_root+'/*/*'))/int(os.environ['BATCH_SIZE']))) # top
        os.environ['TEST_STEPS'] = str(np.ceil(len(glob.glob(test_root+'/*/*'))/int(os.environ['BATCH_SIZE']))) # top
    ###############
    the_best = model.main(layers, filters, train_ds, val_ds, test_ds, class_name=CLASS_NAMES, maxbool=maxbool)
    logging.info(the_best['best_dir']+" SAVED! Best Performance-> Loss:"+str(the_best['best_perform'][0])+", Accuracy:"+str(the_best['best_perform'][1])+", Recall:"+str(the_best['best_perform'][2])+", Precision:"+str(the_best['best_perform'][3]))
    # logging.info("Save_model to: "+the_best['pbpath'])
    logging.info("###################")
    # for l in layers:
    #     for f in filters:
            # the_best = model.main(l, f, train_ds, val_ds, test_ds, class_name=CLASS_NAMES, maxbool=maxbool)
            # logging.info(servable_model.main(the_best['dir'], the_best['pbpath']))
    # while True:
    #     time.sleep(30)
    #     print(" ")
    # plot_history(conv_model.history.history)