import tensorflow as tf
# import sklearn
import numpy as np
import tensorflow.keras.layers as ly

### Models

def BaseModel():
  inputs = {
      "image": tf.keras.Input((FLAGS.img_size, FLAGS.img_size, FLAGS.channels), name="image"),
      "degree": tf.keras.Input([1], name="degree"),
  }
  e = inputs["degree"]
  e = ly.Embedding(FLAGS.n_degree+1, FLAGS.img_size*FLAGS.img_size*FLAGS.channels,
                    input_length=1)(e)
  e = ly.Reshape((FLAGS.img_size, FLAGS.img_size, FLAGS.channels))(e)

  x = inputs["image"]
  x = ly.Add()([x, e])
  x = ly.Conv2D(64, 7, padding="same")(x)
  x = ly.ReLU()(x)
  for filters in [128, 256, 512]:
      x = ly.Conv2D(filters, 3, padding="same")(x)
      x = ly.BatchNormalization()(x)
      x = ly.ReLU()(x)
      x = ly.MaxPool2D()(x)
  x = ly.Conv2D(1024, 3, padding="same")(x)
  x = ly.BatchNormalization()(x)
  x = ly.ReLU()(x)
  x = ly.GlobalAveragePooling2D()(x)
  x = ly.Dense(FLAGS.n_label)(x)
  x = ly.Activation("softmax", dtype="float32")(x)
  return tf.keras.Model(inputs=inputs, outputs=x)

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

def ConcatConv(conv_count, first_filter, 
                kernel_size, strides, 
                input_shape, degree_num, 
                label_num, maxpool=False):
  inputs = {
        'image': tf.keras.Input(input_shape, name='image'),
        'degree': tf.keras.Input([degree_num], name='degree')
    }
  x = inputs['image']
  d = inputs['degree']
  for i in range(conv_count):
    x = ly.Conv2D(first_filter*(i+1), kernel_size, (strides, strides), 
                  padding='same', name='conv'+str(i+1), activation=tf.nn.relu)(x)
    x = ly.BatchNormalization()(x)
    if maxpool and i < conv_count - 1:
        x = ly.MaxPool2D()(x)
  x = ly.GlobalAveragePooling2D(name='GAP')(x)
  # for num_units in hparams.num_fc_units:
  #   x = tf.keras.layers.Dense(num_units, activation='relu')(x)
  x = ly.Concatenate()([x, d])
  x = ly.Dense(label_num, name='dense_logits')(x)
  x = ly.Activation('softmax', dtype='float32', name='predictions')(x)
  return tf.keras.Model(inputs=inputs, outputs=x)

def SomeConv(conv_count, first_filter, kernel_size, strides, input_shape, degree_num, maxpool=False):
  input_image = tf.keras.Input(input_shape, name='image')
  # inputs = tf.keras.Input(shape=(), dtype=tf.string, name='input')
  # x = ly.Lambda(lambda img : tf.map_fn(decode_img, img, dtype=tf.float32))(inputs)
  # # x = tf.map_fn(decode_img, inputs, dtype=tf.float32)
  x = input_image
  for i in range(conv_count):
    x = ly.Conv2D(first_filter*(i+1), kernel_size, (strides, strides), 
                  padding='same', name='conv'+str(i+1), activation=tf.nn.relu)(x)
    x = ly.BatchNormalization()(x)
    if maxpool and i < conv_count - 1:
      x = ly.MaxPool2D()(x)
  x = ly.GlobalAveragePooling2D(name='GAP')(x)
  # for num_units in hparams.num_fc_units:
  #   x = tf.keras.layers.Dense(num_units, activation='relu')(x)
  x = ly.Dense(degree_num, name='dense_logits')(x)
  x = ly.Activation('softmax', dtype='float32', name='predictions')(x)
  return tf.keras.Model(inputs=input_image, outputs=x)

def TransferModel(model, input_shape, label_num, aug=None, output_bias=None):
  inputs = {
        'image': tf.keras.Input(input_shape, name='image'),
        # 'degree': tf.keras.Input([degree_num], name='degree'),
  }
  x = inputs['image']
  # d = inputs['degree']

  # d = ly.Embedding(degree_num+1, input_shape[0]*input_shape[1]*input_shape[2],
  #                    input_length=1)(d)
  # d = ly.Reshape(input_shape)(d)
  # x = ly.Add()([x, d])

  if model == "InceptionV3":
    model_body = tf.keras.applications.InceptionV3(input_shape=input_shape, 
      include_top=False, weights=None)
  elif model == "MobileNetV2":
    model_body = tf.keras.applications.MobileNetV2(input_shape=input_shape, 
      include_top=False, weights=None)
  elif model == "MobileNet":
    model_body = tf.keras.applications.MobileNet(input_shape=input_shape, 
      include_top=False, weights=None)
  elif model == "ResNet50V2":
    model_body = tf.keras.applications.ResNet50V2(input_shape=input_shape, 
      include_top=False, weights=None)
  elif model == "ResNet50":
    model_body = tf.keras.applications.ResNet50(input_shape=input_shape, 
      include_top=False, weights=None)
  elif model == "DenseNet121":
    model_body = tf.keras.applications.DenseNet121(input_shape=input_shape, 
      include_top=False, weights=None)
  elif model == "InceptionResNetV2":
    model_body = tf.keras.applications.InceptionResNetV2(input_shape=input_shape, 
      include_top=False, weights=None)
  elif model == "VGG16":
    model_body = tf.keras.applications.VGG16(input_shape=input_shape, 
      include_top=False, weights=None)
  elif model == "Xception":
    model_body = tf.keras.applications.Xception(input_shape=input_shape, 
      include_top=False, weights=None)

  for layer in model_body.layers:
    layer.trainable = True

  # embedded = embedding(d)
  # x = ly.Concatenate()([x, embedded])
  # x = embedded * x
  if aug != None:
    x = aug(x)
  x = model_body(x)
  x = ly.GlobalAveragePooling2D()(x)
  x = ly.Dense(128, name='dense_128')(x)
  x = ly.Activation('relu', name='act_128')(x)
  # x = ly.Add()([x, d])
  # x = ly.Concatenate()([x, d])
  if output_bias != None:
    x = ly.Dense(label_num, name='dense_logits', bias_initializer=output_bias)(x)
  else:
    x = ly.Dense(label_num, name='dense_logits')(x)
  x = ly.Activation('softmax', dtype='float32', name='predictions')(x)

  return tf.keras.Model(inputs=inputs, outputs=x)


### Custom Metrics

# def recall_m(y_true, y_pred):
#         true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
#         possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
#         recall = true_positives / (possible_positives + K.epsilon())
#         return recall

# def precision_m(y_true, y_pred):
#         true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
#         predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
#         precision = true_positives / (predicted_positives + K.epsilon())
#         return precision

# def f1_m(y_true, y_pred):
#     precision = precision_m(y_true, y_pred)
#     recall = recall_m(y_true, y_pred)
#     return 2*((precision*recall)/(precision+recall+K.epsilon()))

# Recall callback to find metrics at epoch end, works in binary & multi
# to call: Recall(x, y)
class Recall(tf.keras.callbacks.Callback):
    def __init__(self, x, y):
        self.x = x
        self.y = y if (y.ndim == 1 or y.shape[1] == 1) else np.argmax(y, axis=1)
        self.reports = []

    def on_epoch_end(self, epoch, logs={}):
        y_hat = np.asarray(self.model.predict(self.x))
        y_hat = np.where(y_hat > 0.5, 1, 0) if (y.ndim == 1 or y_hat.shape[1] == 1)  else np.argmax(y_hat, axis=1)
        report = classification_report(self.y,y_hat,output_dict=True)
        self.reports.append(report)
        return

    # Utility method
    def get(self, metrics, of_class):
        return [report[str(of_class)][metrics] for report in self.reports]

class CategoricalTruePositives(tf.keras.metrics.Metric): # only used in binary classification

    def __init__(self, name='categorical_true_positives', **kwargs):
      super(CategoricalTruePositives, self).__init__(name=name, **kwargs)
      self.true_positives = self.add_weight(name='tp', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
      y_pred = tf.reshape(tf.argmax(y_pred, axis=1), shape=(-1, 1))
      values = tf.cast(y_true, 'int32') == tf.cast(y_pred, 'int32')
      values = tf.cast(values, 'float32')
      if sample_weight is not None:
        sample_weight = tf.cast(sample_weight, 'float32')
        values = tf.multiply(values, sample_weight)
      self.true_positives.assign_add(tf.reduce_sum(values))

    def result(self):
      return self.true_positives

    def reset_states(self):
      # The state of the metric will be reset at the start of each epoch.
      self.true_positives.assign(0.)

### Tensorboard Extensions

# def plot_confusion_matrix(cm, class_names):

#     """
#     Returns a matplotlib figure containing the plotted confusion matrix.

#     Args:
#         cm (array, shape = [n, n]): a confusion matrix of integer classes
#         class_names (array, shape = [n]): String names of the integer classes
#     """
#     figure = plt.figure(figsize=(8, 8))
#     plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
#     plt.title("Confusion matrix")
#     plt.colorbar()
#     tick_marks = np.arange(len(class_names))
#     plt.xticks(tick_marks, class_names, rotation=45)
#     plt.yticks(tick_marks, class_names)

#     # Normalize the confusion matrix.
#     cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)

#     # Use white text if squares are dark; otherwise black.
#     threshold = cm.max() / 2.
#     for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
#         color = "white" if cm[i, j] > threshold else "black"
#         plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)

#     plt.tight_layout()
#     plt.ylabel('True label')
#     plt.xlabel('Predicted label')
#     return figure

# def log_confusion_matrix(epoch, logs):
#     # Use the model to predict the values from the validation dataset.
#     test_pred_raw = model.predict(test_images)
#     test_pred = np.argmax(test_pred_raw, axis=1)

#     # Calculate the confusion matrix.
#     cm = sklearn.metrics.confusion_matrix(test_labels, test_pred)
#     # Log the confusion matrix as an image summary.
#     figure = plot_confusion_matrix(cm, class_names=DEGREE_CLASS_LIST)
#     cm_image = plot_to_image(figure)

#     # Log the confusion matrix as an image summary.
#     with file_writer_cm.as_default():
#         tf.summary.image("Confusion Matrix", cm_image, step=epoch)