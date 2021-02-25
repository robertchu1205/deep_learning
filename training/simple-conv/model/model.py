from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
# from tensorflow.keras import backend as K
import time
import datetime
import os
import glob
import numpy as np
import logging
###
# for path in train_dir[:3]:
#     display.display(Image.open(str(path)))
# ####Load Images by keras preprocessing####
# ####Downside: Slow, lack of fine-grained control, not well integrated with tf####
# # The 1./255 is to convert from uint8 to float32 in range [0,1].
# image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
# train_data_gen = image_generator.flow_from_directory(directory=r'/tf/robertnb/M1/',
#                                                      batch_size=BATCH_SIZE,
#                                                      color_mode="rgb",
#                                                      shuffle=True,
#                                                      target_size=(IMG_HEIGHT, IMG_WIDTH),
#                                                      classes = CLASS_NAMES,
#                                                      class_mode='categorical'
#                                                      )
###

def decode_img(img_string):
    # convert the compressed string to a 3D uint8 tensor
    # img_string = tf.io.decode_base64(img_string)
    img = tf.io.decode_png(img_string, channels=int(os.environ['CHANNELS']), dtype=tf.dtypes.uint8)
    # img = tf.io.decode_image(img_string, channels=int(os.environ['CHANNELS']), expand_animations=False)
    # Use `convert_image_dtype` to convert to floats in the [0,1] range.
    img = tf.image.convert_image_dtype(img, tf.float32)
    # resize the image to the desired size.
    img = tf.image.resize(img, [int(os.environ['IMG_HEIGHT']), int(os.environ['IMG_WIDTH'])])
    return img

class MyB64ToArrayLayer(tf.keras.layers.Layer):
  def __init__(self, HEIGHT, WIDTH, CHANNEL):
    self.HEIGHT = HEIGHT
    self.WIDTH = WIDTH
    self.CHANNEL = CHANNEL
    super(MyB64ToArrayLayer, self).__init__()

#   def build(self, input_shape):
#     self.kernel = self.add_variable("kernel",
#                                     shape=[int(input_shape[-1]),
#                                            self.HEIGHT,
#                                            self.WIDTH,
#                                            self.CHANNEL])

  def call(self, input):
      decode_img(input[-1][0])
  def compute_output_shape(self, input_shape):
        return (input_shape[0], self.HEIGHT, self.WIDTH, self.CHANNEL)

def show_batch(image_batch, label_batch):
  plt.figure(figsize=(10,10))
  for n in range(25):
    ax = plt.subplot(5,5,n+1)
    plt.imshow(image_batch[n])
    plt.title(CLASS_NAMES[np.argmax(label_batch[n])])
    plt.axis('off')

def seq_simpleConv(conv_count, first_filter, input_tensor, CLASS_NAMES, maxbool=False):
    ly = tf.keras.layers
    model = tf.keras.models.Sequential()
    model.add(ly.InputLayer(input_shape=input_tensor, name='input'))
    # model.add(MyB64ToArrayLayer(int(os.environ['IMG_HEIGHT']), int(os.environ['IMG_WIDTH']),int(os.environ['CHANNELS'])))
    for i in range(conv_count):
        model.add(ly.Conv2D(first_filter*(i+1),(3,3),2,padding='same',name='conv'+str(i)))
        model.add(ly.BatchNormalization())
        model.add(ly.Activation('relu'))
        if maxbool:
            model.add(ly.MaxPooling2D())
    model.add(ly.GlobalAveragePooling2D(name='GAP'))
    model.add(ly.Dense(len(CLASS_NAMES),activation='softmax',name='DENSE'))
    return model

def model_simpleConv(conv_count, first_filter, input_tensor, CLASS_NAMES, maxbool=False):
    ly = tf.keras.layers
    inputs = tf.keras.Input(shape=input_tensor, name='input')
    # inputs = tf.keras.Input(shape=(), dtype=tf.string, name='input')
    # x = ly.Lambda(lambda img : tf.map_fn(decode_img, img, dtype=tf.float32))(inputs)
    # x = tf.map_fn(decode_img, inputs, dtype=tf.float32)
    for i in range(conv_count):
        if i==0: 
            x = ly.Conv2D(first_filter*(i+1),(3,3),2,padding='same',name='conv'+str(i), activation=tf.nn.relu)(inputs)
        else:
            x = ly.Conv2D(first_filter*(i+1),(3,3),2,padding='same',name='conv'+str(i), activation=tf.nn.relu)(x)
        # x = ly.Conv2D(first_filter*(i+1),(3,3),2,padding='same',name='conv'+str(i), activation=tf.nn.relu)(x)
        x = ly.BatchNormalization()(x)
        if maxbool:
            x = ly.MaxPooling2D()(x)
    x = ly.GlobalAveragePooling2D(name='GAP')(x)
    outputs = ly.Dense(len(CLASS_NAMES),activation=tf.nn.softmax,name='DENSE')(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

# def recall_score(y_true, y_pred):
#     y_true, y_pred = K.argmax(y_true), K.argmax(y_pred)
#     true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
#     all_label_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
#     recall = K.cast(true_positives, tf.float32) / \
#         (K.cast(all_label_positives, tf.float32) + tf.constant(K.epsilon()))
#     return recall


# def precision_score(y_true, y_pred):
#     y_true, y_pred = K.argmax(y_true), K.argmax(y_pred)
#     true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
#     predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
#     precision = K.cast(true_positives, tf.float32) / \
#         (K.cast(predicted_positives, tf.float32) + tf.constant(K.epsilon()))
#     return precision

def rm_useless_h5(dirs,test_ds,pbpath):
    best_perform = [100, 0, 0, 0]
    the_best = {}
    for d in dirs:
        try:
            loaded_model = tf.keras.models.load_model(d,compile=True)
            # loaded_model.compile('adam','categorical_crossentropy',['acc', tf.keras.metrics.Recall(), tf.keras.metrics.Precision()])
            perform = loaded_model.evaluate(test_ds, steps=float(os.environ['TEST_STEPS']))
            # logging.info("directory: "+d+", Loss:"+str(perform[0])+", Accuracy:"+str(perform[1])+", Recall:"+str(perform[2])+", Precision:"+str(perform[3]))
            if best_perform[0]>perform[0] and best_perform[1]<=perform[1] and (best_perform[2]<perform[2] or best_perform[3]<perform[3]):
                best_perform = perform
                the_best.update({'best_dir':d})
        except Exception as e:
            logging.info("Incomplete h5 file:{d}, Exception says:{e}".format(d=d,e=e))

    the_best.update({'best_perform':best_perform})
    for d in dirs:
        if d is not the_best['best_dir']:
            # logging.info("directory: "+d+"-> Deleted!")
            os.remove(d)
        # else:
        #     loaded_model = tf.keras.models.load_model(d,compile=True)
        #     tf.saved_model.save(loaded_model, pbpath)
        #     the_best.update({'pbpath':pbpath})
    return the_best

def main(conv_count, first_filter, train_ds, val_ds, test_ds, class_name, maxbool):
    logging.info("conv_count: "+str(conv_count)+ \
                ", first_filter: "+str(first_filter)+ \
                ", IMG_HEIGHT&WIDTH: "+os.environ['IMG_HEIGHT']+ \
                ", MaxBool2D in every layer: "+str(maxbool)+" -> Training Started!")

    # # image_batch, label_batch = next(iter(train_ds))
    # for image, label in train_ds.take(1):
    #     # show_batch(image.numpy(),label.numpy())
    #     # image, label is tf.Tensor
    #     print("Image: ", image.numpy())
    #     print("Label: ", label.numpy())

    mirrored_strategy = tf.distribute.MirroredStrategy(devices=["/gpu:0", "/gpu:1"], \
                        cross_device_ops=tf.distribute.NcclAllReduce())
    filepath='/'+os.environ['MODEL_DIR']+'/conv'+str(conv_count)+'_filter'+str(first_filter)+"_mp"+str(maxbool)
    if not os.path.exists(filepath):
        os.makedirs(filepath)
    # with tf.device('/gpu:1'):
    with mirrored_strategy.scope():
        conv_model = model_simpleConv(conv_count, first_filter, (int(os.environ['IMG_HEIGHT']),int(os.environ['IMG_WIDTH']),int(os.environ['CHANNELS'])), class_name, maxbool)
        conv_model.summary()
        check_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath= filepath+'/conv_'+str(conv_count)+'_ff_'+str(first_filter)+'.{epoch:02d}-{val_loss:.2f}.h5',
            save_best_only=False, # Save all compare it later
            monitor='val_loss',
            mode='auto',
            verbose=1
        )
        tb_callback = tf.keras.callbacks.TensorBoard(
            log_dir=filepath+'/tb_log/fit/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),
            histogram_freq=1
        )
        conv_model.compile('adam','categorical_crossentropy',['acc', tf.keras.metrics.Recall(), tf.keras.metrics.Precision()])
        conv_model.fit(train_ds, epochs=int(os.environ['EPOCH']), validation_data=val_ds, steps_per_epoch=float(os.environ['STEPS_PER_EPOCH']), validation_steps=float(os.environ['VALIDATION_STEPS']),callbacks=[check_callback,tb_callback],verbose=2)
    the_best = rm_useless_h5(glob.glob(filepath+'/*.h5'),test_ds,filepath+'/'+ str(int(time.time())))
    return the_best
