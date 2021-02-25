import os
import tensorflow as tf 
import tensorflow_hub as hub
import glob
import argparse
import sqlite3

parser = argparse.ArgumentParser(description="max_dim of resized images, batch_size")
parser.add_argument("-l", "--LABEL", dest="LABEL", default="OK", help="target label to tfrecord")
parser.add_argument("-m", "--max_dim", dest="MAX_DIM", default="96", help="target shape of resized images")
parser.add_argument("-b", "--batch_size", dest="BATCH_SIZE", default="32", help="batch_size for style transfer")
parser.add_argument("-s", "--all_style", dest="ALL_STYLE", default="0", help="0 is False; 1,2,... is True for representing 1 ori image to all style images")
args = parser.parse_args()
MAX_DIM = int(args.MAX_DIM)
BATCH_SIZE = int(args.BATCH_SIZE)
ALL_STYLE = bool(int(args.ALL_STYLE))
LABEL = args.LABEL
# os.environ['http_proxy'] = os.environ['http_proxy']
# os.environ['https_proxy'] = os.environ['https_proxy']
os.environ["TFHUB_CACHE_DIR"] = '/tf/robertnb/training/style-transfer/style_modules'
hub_module = hub.load('https://hub.tensorflow.google.cn/google/magenta/arbitrary-image-stylization-v1-256/2')
# hub_module = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')
target_cap_dir = f'/data/aoi-wzs-p1-dip-fa-nvidia/label_heatsink_screw/preprocessed/stylized_sh/{LABEL}/'
# style_path = tf.keras.utils.get_file('/tf/robertnb/style_image/kandinsky5.jpg','https://storage.googleapis.com/download.tensorflow.org/example_images/Vassily_Kandinsky%2C_1913_-_Composition_7.jpg')
style_image = '/data/aoi-wzs-p1-dip-fa-nvidia/style_image/*'
styles_fnames = sorted(glob.glob(style_image))
styles_num = len(styles_fnames)

# cap_dir = '/tf/robertnb/Model-2-AllTime/OK/180/*'
# caps_fnames = sorted(glob.glob(cap_dir))
conn = sqlite3.connect('/tf/robertnb/p1-dip-metadata.db')
c = conn.cursor()
caps_fnames = c.execute(f"""select path from metadata 
                            where
                                ( component_class='heat_sink' or
                                component_class='screw') and 
                                label='{LABEL}' and
                                date < 20200503 and 
                                path like '%/image/%'""").fetchall()
caps_fnames = [cf for (cf,) in caps_fnames]

# if BATCH_SIZE / 2 is not int: # BATCH_SIZE / 2 -> 2.0
#     sys.exit("BATCH_SIZE needs to be 2's Multiple")

def parse_img(file_path):
    img = tf.io.read_file(file_path)
    img = tf.io.decode_image(img, channels=3, dtype=tf.dtypes.float32, expand_animations = False)
    img = tf.image.resize_with_pad(img, MAX_DIM, MAX_DIM)
    # img = tf.image.decode_image(img, channels=3) # ValueError: 'images' contains no shape.
    # img = tf.image.convert_image_dtype(img, tf.float32)
    # img = tf.image.resize(img, (MAX_DIM,MAX_DIM))
    # img = img[tf.newaxis, :]
    return img

def parse_style_img(file_path):
    img = tf.io.read_file(file_path)
    img = tf.io.decode_image(img, channels=3, dtype=tf.dtypes.float32, expand_animations = False)
    img = tf.image.resize_with_pad(img, MAX_DIM, MAX_DIM)
    # img = tf.image.decode_jpeg(img, channels=3)
    # img = tf.image.convert_image_dtype(img, tf.float32)
    # img = tf.image.resize(img, (MAX_DIM,MAX_DIM))
    return img

def process_cap(file_path):
    img = parse_img(file_path)
    return tf.strings.split(file_path, '/')[-1], img

def to_zip_ds(all_styles):
    if all_styles is True:
        style_img_ds = tf.data.Dataset.list_files(style_image, shuffle=False)
    else:
        style_img_ds = tf.data.Dataset.list_files(style_image)
    style_img_ds = style_img_ds.map(parse_style_img, num_parallel_calls=tf.data.experimental.AUTOTUNE).repeat()
    if all_styles is True:
        dup_caps_fnames = []
        for cf in caps_fnames:
            for i in range(styles_num):
                dup_caps_fnames.append(cf)
        cap_img_ds = tf.data.Dataset.list_files(dup_caps_fnames, shuffle=False)
    else:
        cap_img_ds = tf.data.Dataset.list_files(caps_fnames)
    cap_fname_img_ds = cap_img_ds.map(process_cap, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    zipped_ds = tf.data.Dataset.zip((cap_fname_img_ds, style_img_ds))
    if all_styles is True:
        zipped_ds = zipped_ds.batch(styles_num)
    else:
        zipped_ds = zipped_ds.batch(BATCH_SIZE)
    return zipped_ds


if __name__ == "__main__":
    mirrored_strategy = tf.distribute.MirroredStrategy(devices=["/gpu:0", "/gpu:1"], \
                        cross_device_ops=tf.distribute.NcclAllReduce())
    with mirrored_strategy.scope():
        for d in to_zip_ds(ALL_STYLE):
            (fname, cap_img), style_img = d
            stylized_img = hub_module(cap_img, style_img)[0]
            stylized_img = tf.image.convert_image_dtype(stylized_img, tf.uint8)

            for idx, img in enumerate(stylized_img):
                # encoded_img = tf.image.encode_png(img)
                encoded_img = tf.io.encode_jpeg(img)
                if ALL_STYLE is True:
                    new_filename = target_cap_dir+fname[idx].numpy().decode().split('.')[0]+'-'+str(idx)+'.jpg'
                else:
                    new_filename = target_cap_dir+fname[idx].numpy().decode()
                tf.io.write_file(
                    filename= new_filename,
                    contents= encoded_img,
                    name=None
                )
                print(new_filename+' Export!')
