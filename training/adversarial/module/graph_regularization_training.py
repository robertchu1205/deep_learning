import os
import time
import tensorflow as tf
import neural_structured_learning as nsl

from absl import app, flags


ly = tf.keras.layers
FLAGS = flags.FLAGS

# paths
flags.DEFINE_string(
    "table_degree_txt", "/p3/graph_regularization/table_degree.txt", "degree lookup table")
flags.DEFINE_string(
    "table_label_txt", "/p3/graph_regularization/table_label.txt", "label lookup table")
flags.DEFINE_string(
    "db_path", "/tf/robertnb/training/metadata.db", "metadata sqlite db path")
flags.DEFINE_string(
    "data_path", "/p3/graph_regularization/tfrecord/", "tfrecord saved path")
flags.DEFINE_string(
    "base_path", "/p3/graph_regularization/results/default/", "model checkpoint base path")
flags.DEFINE_string(
    "model_export_path", "/p3/graph_regularization/results/default/model.h5", "model export path")

# data
flags.DEFINE_integer("n_degree", 4, "total number of distinct degree")
flags.DEFINE_integer("n_label", 2, "total number of distinct label")
flags.DEFINE_integer("channels", 3, "image channels")
flags.DEFINE_integer("embedding_img_size", 75,
                     "image size for generating embedding")
flags.DEFINE_integer("img_size", 32, "image size for training")

# build graph
flags.DEFINE_float("similarity_threshold", 0.99,
                   "similarity threshold for building graph")

# dataset
flags.DEFINE_integer("shuffle_buffer", 12000, None)
flags.DEFINE_integer("valid_size", 2000, None)
flags.DEFINE_integer("batch_size", 1024, None)

# training
flags.DEFINE_integer("epochs", 50, None)
flags.DEFINE_integer("steps_per_epoch", 10, None) # better use max gpu, increase to see what this gonna be
flags.DEFINE_integer("verbose", 0, None)
flags.DEFINE_boolean("mixed_precision", False, None)
flags.DEFINE_string(
    "run_id", f"adv_graph_model_nbr_{time.time()}", None)
flags.DEFINE_boolean(
    "previous_dataset", True, None)

# graph & adversarial
flags.DEFINE_integer(
    "max_nbrs", 3, "max number of neighbor examples while building training tfrecord")
flags.DEFINE_integer(
    "n_nbr", 3, "max number of neighbor examples while parsing tfrecord")
flags.DEFINE_float("g_multiplier", 0.01, "weight factor for graph loss term")
flags.DEFINE_float("adv_multiplier", 0.2,
                   "weight factor for adversarial loss term")
flags.DEFINE_float("adv_step_size", 0.2,
                   "step size for finding adversarial example (0.01-0.2)")


table_degree = None
table_label = None


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def create_embedding_example(image_embedding, record_id):
    features = {
        "id": _bytes_feature(record_id),
        "embedding": _float_feature(image_embedding),
    }
    return tf.train.Example(features=tf.train.Features(feature=features))


def parse_metadata_for_embedding(path):
    image = tf.io.read_file(path)
    image = tf.io.decode_jpeg(image, FLAGS.channels)
    image = tf.image.resize_with_pad(
        image, FLAGS.embedding_img_size, FLAGS.embedding_img_size, tf.image.ResizeMethod.BILINEAR)
    features = {
        "path": path,
        "image": image,
    }
    return features


def create_embedding_tfrecord(dataset, output_path):
    extractor = tf.keras.applications.InceptionV3(
        include_top=False,
        input_shape=(FLAGS.embedding_img_size,
                     FLAGS.embedding_img_size, FLAGS.channels))
    with tf.io.TFRecordWriter(output_path) as writer:
        for features in dataset:
            embeddings = extractor(features["image"])
            embeddings = tf.squeeze(embeddings) # remove 1 in dimensions e.g. (64, 1, 1, 1024) -> (64, 1024)
            for embedding, record_id in zip(embeddings, features["path"]):
                example = create_embedding_example(
                    embedding.numpy().tolist(), [record_id.numpy()])
                writer.write(example.SerializeToString())


def generate_image_embeddings_embedding_by_label_component(label, component, export_path):
    dataset = tf.data.experimental.SqlDataset(
        "sqlite", FLAGS.db_path,
        f"""select path from metadata
        where
            test_label = '{label}' and
            component = '{component}' and 
            degree >= 0 and 
            extension = 'png'
            ORDER BY RANDOM() LIMIT 10000
        """, (tf.string))
    dataset = dataset.shuffle(FLAGS.shuffle_buffer).map(
        parse_metadata_for_embedding, tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(FLAGS.batch_size)

    create_embedding_tfrecord(dataset, export_path)


def parse_metadata_and_serialize_example(path, degree, label):
    image = tf.io.read_file(path)
    degree = table_degree.lookup(degree)
    label = table_label.lookup(label)

    features = {
        "path": _bytes_feature([path.numpy()]),
        "image": _bytes_feature([image.numpy()]),
        "degree": _int64_feature([degree.numpy()]),
        "label": _int64_feature([label.numpy()]),
    }
    example_proto = tf.train.Example(
        features=tf.train.Features(feature=features))
    return example_proto.SerializeToString()


def tf_serialize_example(path, degree, label):
    tf_string = tf.py_function(
        parse_metadata_and_serialize_example,
        (path, degree, label),
        tf.string)
    return tf.reshape(tf_string, ())


def generate_training_tfrecord_by_label_component(label, component, export_path):
    dataset = tf.data.experimental.SqlDataset(
        "sqlite", FLAGS.db_path,
        f"""select path, degree, test_label from metadata
        where
            test_label = '{label}' and
            component = '{component}' and 
            degree >= 0 and 
            extension = 'png'
            ORDER BY RANDOM() LIMIT 10000
        """, (tf.string, tf.string, tf.string))
    dataset = dataset.map(tf_serialize_example, tf.data.experimental.AUTOTUNE)
    writer = tf.data.experimental.TFRecordWriter(export_path)
    writer.write(dataset)


def parse_image(image):
    # image = tf.io.decode_image(image, FLAGS.channels, expand_animations = False) # dtype=tf.dtypes.float32
    image = tf.io.decode_png(image, FLAGS.channels)
    image = tf.image.resize_with_pad(
        image, FLAGS.img_size, FLAGS.img_size, tf.image.ResizeMethod.BILINEAR)
    return image


def parse_example(example_proto): # 1 -> 2, 1 -> 3, (X) 2 -> 1
    feature_spec = {
        "degree": tf.io.FixedLenFeature([], tf.int64),
        "image": tf.io.FixedLenFeature([], tf.string),
        "label": tf.io.FixedLenFeature([], tf.int64),
    }

    empty_png = tf.zeros(
        (FLAGS.img_size, FLAGS.img_size, FLAGS.channels), tf.uint8)
    empty_png = tf.io.encode_jpeg(empty_png)

    for i in range(FLAGS.n_nbr):
        nbr_feature_key_image = f"NL_nbr_{i}_image"
        nbr_feature_key_degree = f"NL_nbr_{i}_degree"
        nbr_weight_key = f"NL_nbr_{i}_weight"

        feature_spec[nbr_feature_key_image] = tf.io.FixedLenFeature(
            [], tf.string, empty_png)
        feature_spec[nbr_feature_key_degree] = tf.io.FixedLenFeature(
            [1], tf.int64, tf.constant([0], tf.int64))
        feature_spec[nbr_weight_key] = tf.io.FixedLenFeature(
            [1], tf.float32, tf.constant([0.0]))

    features = tf.io.parse_single_example(example_proto, feature_spec)

    features["image"] = parse_image(features["image"])
    features["label"] = tf.one_hot(features["label"], FLAGS.n_label)
    for i in range(FLAGS.max_nbrs):
        nbr_feature_key = f"NL_nbr_{i}_image"
        features[nbr_feature_key] = parse_image(features[nbr_feature_key])

    label = features["label"]
    return features, label


def create_training_dataset(tfrecord_paths):
    # the repeat here is essential, because if we don't repeat,
    # sample_from_dataset will drain small dataset and won't balance datasets
    datasets = [tf.data.TFRecordDataset(x).repeat() for x in tfrecord_paths]

    dataset = tf.data.experimental.sample_from_datasets(datasets)
    dataset = dataset.map(parse_example, tf.data.experimental.AUTOTUNE)

    train_ds = dataset.skip(FLAGS.valid_size).shuffle(
        FLAGS.shuffle_buffer).batch(FLAGS.batch_size).prefetch(tf.data.experimental.AUTOTUNE)
    valid_ds = dataset.take(FLAGS.valid_size).shuffle(
        FLAGS.shuffle_buffer).batch(FLAGS.batch_size).prefetch(tf.data.experimental.AUTOTUNE)

    return train_ds, valid_ds


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
    x = ly.Add()([x, e]) # value added, same shape
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


def build_adv_graph_model(base_model): # must be graph then adv
    adv_config = nsl.configs.make_adv_reg_config(
        multiplier=FLAGS.adv_multiplier,
        adv_step_size=FLAGS.adv_step_size)
    graph_reg_config = nsl.configs.make_graph_reg_config(
        max_neighbors=FLAGS.n_nbr,
        multiplier=FLAGS.g_multiplier)

    graph_reg_model = nsl.keras.GraphRegularization(
        base_model, graph_reg_config)
    adv_graph_model = nsl.keras.AdversarialRegularization(
        graph_reg_model, label_keys=["label"], adv_config=adv_config)

    return adv_graph_model

def train_previous_tfrecords():
    tfrecord_paths = []
    for label in ["NG-MoreComp", "NG-NoneComp", "NG-UpsideDown", "NG-OutsidePosition", "NG-InversePolarity", "OK"]:
        for component in ["AluCap", "ElecCap"]:
            export_path = os.path.join(
                FLAGS.data_path, f"{label}-{component}.tfrecord")
            tfrecord_paths.append(export_path)
    training_tfrecord_paths = []
    for tfrecord_path in tfrecord_paths:
        nbr_tfrecord_path = os.path.join(
            FLAGS.data_path, f"nbr-099-{os.path.basename(tfrecord_path)}") # target dir after pack
        training_tfrecord_paths.append(nbr_tfrecord_path)
    # prepare data for training
    train_ds, valid_ds = create_training_dataset(training_tfrecord_paths)
    # start training
    training(train_ds, valid_ds)

def training(train_ds, valid_ds):
    # build model
    if FLAGS.mixed_precision:
        policy = tf.keras.mixed_precision.experimental.Policy("mixed_float16")
        tf.keras.mixed_precision.experimental.set_policy(policy)

    base_model = BaseModel()
    model = build_adv_graph_model(base_model)

    # training model
    metrics = [
        tf.keras.metrics.CategoricalAccuracy(name='accuracy'),
        tf.keras.metrics.Precision(name="precision/ok", class_id=1),
        tf.keras.metrics.Precision(name="precision/ng", class_id=0),
        tf.keras.metrics.Recall(name="recall/ok", class_id=1),
        tf.keras.metrics.Recall(name="recall/ng", class_id=0),
    ]

    run_dir = os.path.join(FLAGS.base_path, FLAGS.run_id)
    callbacks = [
        tf.keras.callbacks.TensorBoard(
            run_dir, write_graph=False, profile_batch=0),
        tf.keras.callbacks.ModelCheckpoint(
            run_dir, monitor="recall/ng", save_best_only=True, mode="max") # save_weights_only=True
    ]
    model.compile(
        "adam", tf.keras.losses.CategoricalCrossentropy(), metrics)

    model.fit(train_ds, validation_data=valid_ds, steps_per_epoch=FLAGS.steps_per_epoch,
              epochs=FLAGS.epochs, verbose=FLAGS.verbose, callbacks=callbacks)

    model.base_model.base_model.save_weights(FLAGS.model_export_path)

def main(_):
    # prepare lookup table
    global table_degree
    global table_label
    table_degree = tf.lookup.StaticHashTable(tf.lookup.TextFileInitializer(
        FLAGS.table_degree_txt,
        tf.string, tf.lookup.TextFileIndex.WHOLE_LINE,
        tf.int64, tf.lookup.TextFileIndex.LINE_NUMBER
    ), FLAGS.n_degree)
    table_label = tf.lookup.StaticHashTable(tf.lookup.TextFileInitializer(
        FLAGS.table_label_txt,
        tf.string, 0,
        tf.int64, 1, delimiter=" "
    ), -1)

    # while using previously created tfrecords
    if FLAGS.previous_dataset is True:
        train_previous_tfrecords()
        return 0

    # generate_image_embeddings_embedding_by_label_component
    embed_tfrecord_path = []
    for label in ["NG-MoreComp", "NG-NoneComp", "NG-UpsideDown", "NG-OutsidePosition", "NG-InversePolarity", "OK"]:
        for component in ["AluCap", "ElecCap"]:
            export_path = os.path.join(
                FLAGS.data_path, f"embeddings-{label}-{component}.tfrecord")
            generate_image_embeddings_embedding_by_label_component(
                label, component, export_path)
            embed_tfrecord_path.append(export_path)

    # build graph from embeddings by label, component
    graph_paths = []
    for path in embed_tfrecord_path:
        graph_path = path.replace("embeddings", "graph").replace(
            ".tfrecord", f"-{FLAGS.similarity_threshold}.tsv")
        graph_paths.append(graph_path)
        nsl.tools.build_graph([path], graph_path,
                              similarity_threshold=FLAGS.similarity_threshold) # only includes larger threshold

    # pull dataset from metadata.db and generate training tfrecords
    tfrecord_paths = []
    for label in ["NG-MoreComp", "NG-NoneComp", "NG-UpsideDown", "NG-OutsidePosition", "NG-InversePolarity", "OK"]:
        for component in ["AluCap", "ElecCap"]:
            export_path = os.path.join(
                FLAGS.data_path, f"{label}-{component}.tfrecord")
            generate_training_tfrecord_by_label_component(
                label, component, export_path)
            tfrecord_paths.append(export_path)

    # augment dataset with neighbors
    training_tfrecord_paths = []
    for tfrecord_path, graph_path in zip(tfrecord_paths, graph_paths):
        nbr_tfrecord_path = os.path.join(
            FLAGS.data_path, f"nbr-099-{os.path.basename(tfrecord_path)}") # target dir after pack
        training_tfrecord_paths.append(nbr_tfrecord_path)
        nsl.tools.pack_nbrs(tfrecord_path, "", graph_path,
                            nbr_tfrecord_path, max_nbrs=FLAGS.max_nbrs, add_undirected_edges=True, id_feature_name="path")

    # prepare data for training
    train_ds, valid_ds = create_training_dataset(training_tfrecord_paths)

    # start training
    training(train_ds, valid_ds)

if __name__ == "__main__":
    app.run(main)