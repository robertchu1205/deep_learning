import tensorflow as tf
from tensorflow.keras import backend as K
tf.compat.v1.disable_eager_execution()
# import argparse
import time, os

h5_path = os.environ['h5_path']
pb_dir = os.environ['pb_dir']
img_size = int(os.environ['img_size'])

def init():
    parser = argparse.ArgumentParser(
            description='provide pretrain model (.h5) directorys and target directory for saved pb model by arg')
    parser.add_argument('-i', '--h5_path', dest='h5_path', help='pretrain model (.h5) file (whole file path)')
    parser.add_argument('-o', '--pb_dir', dest='pb_dir', help='target directory for saved pb model')
    args = parser.parse_args()
    h5_path = args.h5_path
    pb_dir = args.pb_dir

    DEGREE_CLASS_LIST = ['0', '180', '270', '90']
    DEGREE_TABLE = tf.constant(list(range(len(DEGREE_CLASS_LIST))), dtype=tf.int64)
    degree_lookup = tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(DEGREE_CLASS_LIST, DEGREE_TABLE), -1)

@tf.function
def f(self, receiver_tensors):
    string_tensor = receiver_tensors['image']
    with tf.device("/cpu:0"):
        input_string = tf.nest.map_structure(tf.stop_gradient, 
                tf.map_fn(parse_img, string_tensor, dtype=tf.float32))
    return input_string

def To_servable_model(H5_PATH, SAVED_PB_PATH):
    SAVED_PB_PATH = SAVED_PB_PATH + str(int(time.time())) 
    K.set_learning_phase(0)
    # tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session())
    loaded_model = tf.keras.models.load_model(H5_PATH)
    receiver_tensors = { #json request input
        'image': tf.compat.v1.placeholder(dtype=tf.string,shape=[None]),
        # 'degree': tf.compat.v1.placeholder(dtype=tf.string,shape=[None]),
    }
    features = {
        'image': receiver_tensors['image'],
        # 'degree': receiver_tensors['degree'],
    }
    try:
        features['image'] = f(receiver_tensors)
        # features['image'] = tf.map_fn(
        #     parse_img,receiver_tensors['image'], 
        #     dtype=tf.float32, back_prop=False) 
        # features['degree'] = tf.map_fn(
        #     parse_degree,receiver_tensors['degree'], 
        #     dtype=tf.float32, back_prop=False) 
    except Exception as e:
        print(e)
        return 'Failed'
    model_softmax = loaded_model(features)
    predictions = {
        'output_node': model_softmax
    }
    builder = tf.compat.v1.saved_model.builder.SavedModelBuilder(SAVED_PB_PATH)
    signature = tf.compat.v1.saved_model.predict_signature_def(inputs=receiver_tensors, outputs=predictions)
    with tf.compat.v1.keras.backend.get_session() as sess:
        try:
            builder.add_meta_graph_and_variables(
                sess=sess,
                tags=[tf.saved_model.SERVING],
                signature_def_map={'classification': signature},
                legacy_init_op=tf.compat.v1.tables_initializer())
        except:
            builder.add_meta_graph_and_variables(
                sess=sess,
                tags=[tf.saved_model.SERVING],
                signature_def_map={'classification': signature})
        builder.save()
    return SAVED_PB_PATH

def parse_img(img):
    img = tf.io.decode_image(img, 3, dtype=tf.dtypes.float32, expand_animations = False)
    img = tf.image.resize_with_pad(img, img_size, img_size)
    # img = tf.cast(img, dtype=tf.float32) / 255.
    # img = tf.image.convert_image_dtype(img, tf.float32)
    return img

def parse_degree(degree):
    degree = tf.strings.as_string(
        tf.math.abs(tf.strings.to_number(degree, out_type=tf.dtypes.int64)))
    degree = degree_lookup.lookup(degree)
    onehot_degree = tf.one_hot(degree, len(DEGREE_CLASS_LIST), dtype='int64')
    onehot_degree = tf.cast(onehot_degree, dtype=tf.float32)
    return onehot_degree

def main(h5path, pbpath):
    try:
        return f'H5: {h5path} export to PB: {To_servable_model(h5path, pbpath)}'
    except Exception as e:
        return f'except while servable model. Error: {e}'

if __name__ == '__main__':
    # init()
    print(main(h5_path, pb_dir))