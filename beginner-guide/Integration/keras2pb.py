import tensorflow as tf
import os
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model
import time
from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import utils
from tensorflow.python.saved_model import tag_constants, signature_constants
from tensorflow.python.saved_model.signature_def_utils_impl import build_signature_def, predict_signature_def
from tensorflow.contrib.session_bundle import exporter
import argparse

# Directory comes from reference right behind compiling python file
# For example: python keras2pb.py -d ./model1.h5 ./model2.h5
parser = argparse.ArgumentParser(description="provide pretrain model (.h5) directorys by arg")
parser.add_argument("-d", "--h5_dir", dest="h5_dir", help="pretrain model (.h5) directorys",nargs='+')
args = parser.parse_args()
dirs = args.h5_dir
MODEL1_H5 = dirs[0] 
MODEL2_H5 = dirs[1]
# Directory comes from Environmental Variable & convert current time to version number
SAVED_PB_PATH = os.environ['SAVED_PB_PATH']+ str(int(time.time())) 
IMAGE_SIZE = 76 #img size corrsponding to training model input

# For freezing trained model
# layer.trainable = False -> NOT WORK
K.set_learning_phase(0) 
m1 = tf.keras.models.load_model(filepath=MODEL1_H5)
m2 = tf.keras.models.load_model(filepath=MODEL2_H5)
# Angle from string input to index to onehot
# CAN NOT put in serving_parse_fn, 
# otherwise the placeholder will be in different graph, even happen Invalid argument 
# ( must feed a value for placeholder tensor )
angletableidx = tf.constant(['0','90','180','270']) 
angletable = tf.contrib.lookup.string_to_index_table_from_tensor(angletableidx)

def serving_parse_fn(image_string):
    image_decode = tf.image.decode_bmp(image_string,channels=3) # client input encoded images to base64
    image_resized = tf.image.resize_images(image_decode,[IMAGE_SIZE,IMAGE_SIZE])
    img = tf.image.convert_image_dtype(image_resized, tf.float32)
    img = img / 255. #can not -0.5 cause it's not in training process
    return img

def preprocess_angle(angle_string):
    angle_index = angletable.lookup(angle_string)
    angle_onehot = tf.one_hot(angle_index,depth=4,dtype=tf.float32)
    return angle_onehot

receiver_tensors = { #json request input
    "image": tf.placeholder(dtype=tf.string,shape=[None]),
    "angle": tf.placeholder(dtype=tf.string,shape=[None]),
}

features = {
    "image": receiver_tensors['image'],
    "angle": receiver_tensors['angle'],
}

# parse input to what we want
features['image'] = tf.map_fn(serving_parse_fn,receiver_tensors['image'], dtype=tf.float32, back_prop=False) 
features['angle'] = tf.map_fn(preprocess_angle,receiver_tensors['angle'], dtype=tf.float32, back_prop=False)

# Model 1 Output -> 0:'NG-LoC',1:'NG-MtC',2:'NG-TH',3:'NG-UD',4:'OK'
m1_output = m1(features['image']) 
# Model 2 Input -> Images & Angle OK Capacitor should be
# Model 2 Output -> 0:'OK', 1:'NG'
m2_output = m2([features['image'],features['angle']]) 

m1_pred_index = tf.argmax(m1_output, 1) #prediction index ##
m1_confidence = tf.reduce_max(m1_output,1) # prediction confidence
m2_pred_index = tf.argmax(m2_output, 1) #prediction index ##
m2_confidence = tf.reduce_max(m2_output,1) # prediction confidence

mapping_m1 = tf.constant(['NG-LoC','NG-MtC','NG-TH','NG-UD','OK'])
phase1table = tf.contrib.lookup.index_to_string_table_from_tensor(mapping_m1)
p1_pred_class_str = phase1table.lookup(m1_pred_index)

mapping_m2 = tf.constant(['OK','NG'])
phase2table = tf.contrib.lookup.index_to_string_table_from_tensor(mapping_m2)
p2_pred_class_str = phase2table.lookup(m2_pred_index)

combined_index = tf.abs(tf.sign(tf.add(m2_pred_index,tf.add(m1_pred_index,-4))))
combined_class_str = phase2table.lookup(combined_index)

predictions = {
    "m2_outcome": p2_pred_class_str,
    "m1_outcome" : p1_pred_class_str,
    "m2_confidence" : m2_confidence,
    "m1_confidence" : m1_confidence,
    "combined_outcome":combined_class_str,
}
builder = tf.saved_model.builder.SavedModelBuilder(SAVED_PB_PATH)
# prediction_signature = (
#         tf.saved_model.signature_def_utils.build_signature_def(
#             inputs={'inputs': model_input},
#             outputs={'output': model_output},
#             method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME))

signature = tf.saved_model.signature_def_utils.predict_signature_def(inputs=receiver_tensors,
                                                                     outputs=predictions)

with K.get_session() as sess:
    builder.add_meta_graph_and_variables(sess=sess,
                                         tags=[tf.saved_model.tag_constants.SERVING],
                                         signature_def_map={'serving_default': signature}, #serving_default is the default of signature_def, otherwise client should alter this setting
                                         legacy_init_op=tf.tables_initializer() #must initialize if using lookup table
                                        )
    builder.save()