import tensorflow as tf
from tensorflow.keras import backend as K
import os
tf.compat.v1.disable_eager_execution()
import argparse
import time
import glob

def To_servable_model(H5_PATH, SAVED_PB_PATH):
    SAVED_PB_PATH = SAVED_PB_PATH + str(int(time.time())) 
    K.set_learning_phase(0)
    # tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session())
    loaded_model = tf.keras.models.load_model(H5_PATH, compile=True)
    # loaded_model.compile('adam','categorical_crossentropy',['acc', tf.keras.metrics.Recall(), tf.keras.metrics.Precision()])
    receiver_tensors = { #json request input
        "string_array": tf.compat.v1.placeholder(dtype=tf.string,shape=[None]),
    }
    features = {
        "string_array": receiver_tensors['string_array'],
    }
    try:
        # elem = next(iter(receiver_tensors['string_array']))
        # d_serving_parse_fn(elem)
        features['string_array'] = tf.map_fn(serving_parse_fn,receiver_tensors['string_array'], dtype=tf.float32, back_prop=False)
        print("serving_parse_fn")
    except:
        features['string_array'] = tf.map_fn(serving_parse_fn,receiver_tensors['string_array'], dtype=tf.float32, back_prop=False) 
    model_softmax = loaded_model(features['string_array'])
    predictions = {
        "output_node": model_softmax
    }
    builder = tf.compat.v1.saved_model.builder.SavedModelBuilder(SAVED_PB_PATH)
    signature = tf.compat.v1.saved_model.predict_signature_def(inputs=receiver_tensors, outputs=predictions)
    with tf.compat.v1.keras.backend.get_session() as sess:
        try:
            builder.add_meta_graph_and_variables(
                sess=sess,
                tags=[tf.saved_model.SERVING],
                signature_def_map={"classification": signature},
                legacy_init_op=tf.compat.v1.tables_initializer())
        except:
            builder.add_meta_graph_and_variables(
                sess=sess,
                tags=[tf.saved_model.SERVING],
                signature_def_map={"classification": signature})
        builder.save()
    return SAVED_PB_PATH

def serving_parse_fn(img):
    img = tf.image.decode_png(img, channels=int(os.environ['CHANNELS']))
    # img = tf.image.decode_image(image_string, channels=int(os.environ['CHANNELS']), expand_animations=True)
    img = tf.image.resize(img, [int(os.environ['IMG_WIDTH']), int(os.environ['IMG_HEIGHT'])])
    img = tf.image.convert_image_dtype(img, tf.float32)
    return img

def d_serving_parse_fn(img):
    # img = tf.compat.as_bytes(img)
    img = tf.io.decode_base64(img)
    img = tf.image.decode_png(img, channels=int(os.environ['CHANNELS']))
    # img = tf.image.decode_image(image_string, channels=int(os.environ['CHANNELS']), expand_animations=True)
    img = tf.image.resize(img, [int(os.environ['IMG_WIDTH']), int(os.environ['IMG_HEIGHT'])])
    img = tf.image.convert_image_dtype(img, tf.float32)
    return img

# @tf.function(input_signature=(tf.TensorSpec(shape=[None,], dtype=tf.string),))
# def serve_load_and_preprocess_image(image_string):
#     # loaded images may need converting to the tensor shape needed for the model  
#     K.set_learning_phase(0)
#     model = tf.keras.models.load_model(os.environ['H5_PATH'], compile=True)
#     # model.compile('adam','categorical_crossentropy',['acc', tf.keras.metrics.Recall(), tf.keras.metrics.Precision()])     
#     loaded_images = tf.map_fn(d_serving_parse_fn, image_string)
#     # try:
#     #     loaded_images = tf.map_fn(d_serving_parse_fn, image_string)
#     # except:
#     #     loaded_images = tf.map_fn(serving_parse_fn, image_string)
#     predictions = model(loaded_images)
#     return predictions

def main(h5path, pbpath):
    try:
        return "H5: {h5path} export to PB: {pbpath} succeed!".format(h5path=h5path,pbpath=To_servable_model(h5path, pbpath))
    except Exception as e:
        return "except while servable model. Error: {error}".format(error=e)

    # serve_load_and_preprocess_image_string = serve_load_and_preprocess_image.get_concrete_function(
    #     image_string=tf.TensorSpec([None,], dtype=tf.string))

    # tf.saved_model.save(
    #     model,
    #     pbpath,
    #     signatures=serve_load_and_preprocess_image_string
    # )

    # check the models give the same output
    # loaded = tf.saved_model.load(MODEL_PATH)
    # loaded_model_predictions = loaded.serve_load_and_preprocess_path(...)
    # np.testing.assert_allclose(trained_model_predictions, loaded_model_predictions, atol=1e-6)

if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description="provide pretrain model (.h5) directorys and target directory for saved pb model by arg")
    # parser.add_argument("-i", "--w_h", dest="w_h", help="weight & height for serving parse", nargs='+')
    # parser.add_argument("-d", "--h5_dir", dest="h5_dir", help="pretrain model (.h5) directorys")
    # parser.add_argument("-p", "--pb_dir", dest="pb_dir", help="target directory for saved pb model")
    # args = parser.parse_args()
    # h5_dir = args.h5_dir
    # pb_dir = args.pb_dir
    # w_h = args.w_h
    # os.environ['IMG_WIDTH'] = w_h[0]
    # os.environ['IMG_HEIGHT'] = w_h[1]
    
    os.environ['CHANNELS'] = '3'
    h5_dir = os.environ['H5_PATH']
    pb_dir = os.environ['PB_PATH']
    h5_dir = glob.glob(h5_dir+'*.h5')[0]

    print(main(h5_dir, pb_dir))
