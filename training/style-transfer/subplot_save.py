import os
import tensorflow as tf 
import tensorflow_hub as hub
import glob
import numpy as np
import matplotlib.pyplot as plt
import argparse
import sys

os.environ['http_proxy'] = os.environ['http_proxy']
os.environ['https_proxy'] = os.environ['https_proxy']
# hub_module = hub.load('https://hub.tensorflow.google.cn/google/magenta/arbitrary-image-stylization-v1-256/2')
hub_module = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')
cap_path = '/tf/robertnb/Model-2-AllTime/OK/180/0907-CN0FPP7FWS20099601SPA00_PT4506_0_NA_NA.png'
style_path = tf.keras.utils.get_file('/tf/robertnb/style_image/kandinsky5.jpg','https://storage.googleapis.com/download.tensorflow.org/example_images/Vassily_Kandinsky%2C_1913_-_Composition_7.jpg')
style_image = '/tf/robertnb/style_image/*'
styles_fnames = sorted(glob.glob(style_image))
styles_num = len(styles_fnames)

parser = argparse.ArgumentParser(description="max_dim of resized images, batch_size")
parser.add_argument("-m", "--max_dim", dest="MAX_DIM", help="max_dim of resized images")
parser.add_argument("-b", "--batch_size", dest="BATCH_SIZE", help="batch_size for subplot whole figure")
args = parser.parse_args()
MAX_DIM = int(args.MAX_DIM)
BATCH_SIZE = int(args.BATCH_SIZE)
# if BATCH_SIZE / 2 is not int: # BATCH_SIZE / 2 -> 2.0
#     sys.exit("BATCH_SIZE needs to be 2's Multiple")

def load_img(path_to_img, max_dim, resize):
    img = tf.io.read_file(path_to_img)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)

    #   SCALE
    #   shape = tf.cast(tf.shape(img)[:-1], tf.float32)
    #   long_dim = max(shape)
    #   scale = max_dim / long_dim
    #   new_shape = tf.cast(shape * scale, tf.int32)
    if resize is True:
        img = tf.image.resize(img, (max_dim,max_dim))

    img = img[tf.newaxis, :]
    return img

if __name__ == "__main__":
    rows = int(BATCH_SIZE / 2)

    for i in range(0, styles_num, BATCH_SIZE):
        style_images = []
        for f in styles_fnames[i:i+BATCH_SIZE]:
            style_images.append(load_img(f, MAX_DIM, True))
        cap_image = load_img(cap_path, MAX_DIM, True)
        stacked_cap = tf.squeeze(tf.stack([cap_image for i in range(len(style_images))],1),axis=0)
        style_images = tf.squeeze(tf.stack(style_images,1),axis=0)
        stylized_image = hub_module(tf.constant(stacked_cap), tf.constant(style_images))
        # stylized_image = hub_module(tf.constant(cap_image), style_images)

        # sharex, sharey 设置为 True 或者 ‘all’ 时，所有子图共享 x 轴或者 y 轴
        # fig, axes = plt.subplots(rows, 2, sharex=True, sharey=True, figsize=(20, 20))
        # for o,ax in zip(stylized_image[0],axes.flatten()):
        #     ax.imshow(o)
        #     ax.axis("off")
        fig = plt.figure(figsize=(14, 7 * rows))
        for idx, si in enumerate(stylized_image[0]):
            ax = plt.subplot(str(rows)+'2'+str(idx))
            ax.imshow(si)
            ax.axis("off")
        fig.tight_layout()

        fig.savefig("/tf/robertnb/styled_images/M-{m}-B-{b}-{i}TO{i2}.png".format(m=MAX_DIM, b=BATCH_SIZE, i=i, i2=i+4))
        break
