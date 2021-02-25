import time, argparse, os
import tensorflow as tf

def model_structure(model, input_shape, label_num):
    inputs = {
        "image": tf.keras.Input(input_shape, name="image"),
    }

    x = inputs["image"]

    if model == "I3":
        model_body = tf.keras.applications.InceptionV3(input_shape=input_shape, 
            include_top=False, weights=None)
    elif model == "M2":
        model_body = tf.keras.applications.MobileNetV2(input_shape=input_shape, 
            include_top=False, weights=None)
    elif model == "M":
        model_body = tf.keras.applications.MobileNet(input_shape=input_shape, 
            include_top=False, weights=None)
    elif model == "R2":
        model_body = tf.keras.applications.ResNet50V2(input_shape=input_shape, 
            include_top=False, weights=None)
    elif model == "R":
        model_body = tf.keras.applications.ResNet50(input_shape=input_shape, 
            include_top=False, weights=None)
    elif model == "D":
        model_body = tf.keras.applications.DenseNet121(input_shape=input_shape, 
            include_top=False, weights=None)
    elif model == "I3V2":
        model_body = tf.keras.applications.InceptionResNetV2(input_shape=input_shape, 
            include_top=False, weights=None)
    elif model == "V":
        model_body = tf.keras.applications.VGG16(input_shape=input_shape, 
            include_top=False, weights=None)
    elif model == "X":
        model_body = tf.keras.applications.Xception(input_shape=input_shape, 
            include_top=False, weights=None)

    # for layer in model_body.layers:
    #     layer.trainable = True

    x = model_body(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(128, name="dense_128")(x)
    x = tf.keras.layers.Dense(label_num, name="dense_logits")(x)
    x = tf.keras.layers.Activation("softmax", dtype="float32", name="predictions")(x)

    return tf.keras.Model(inputs=inputs, outputs=x)

def parse_image(image):
    image = tf.io.decode_jpeg(image,channels=3,dct_method="INTEGER_ACCURATE",try_recover_truncated=True)
    image = tf.cast(image, dtype=tf.dtypes.float32) / 255.0
    image = tf.image.resize_with_pad(image, int(img_size), int(img_size))
    return image

class WrapModel(tf.Module):
    def __init__(self, model):
        self.model = model

        self.table_label_str = tf.lookup.StaticHashTable(tf.lookup.KeyValueTensorInitializer(
            tf.constant([0, 1], dtype=tf.int64), ["NG", "OK"], key_dtype=tf.int64, value_dtype=tf.string
        ), "UNKNOWN")

        self.bool_to_pred_class = tf.lookup.StaticHashTable(tf.lookup.KeyValueTensorInitializer(
            tf.constant([1, 0], dtype=tf.int64), ["OK", "NG"], key_dtype=tf.int64, value_dtype=tf.string
        ), "NG")

    @tf.function(input_signature=[
        tf.TensorSpec(None, dtype=tf.string, name="image"),
        tf.TensorSpec(None, dtype=tf.string, name="component"),
        tf.TensorSpec(None, dtype=tf.string, name="capacity"),
        tf.TensorSpec(None, dtype=tf.string, name="degree"),
        tf.TensorSpec(None, dtype=tf.string, name="voltage"),
        tf.TensorSpec(None, dtype=tf.string, name="SN"),
    ])
    def serve(self, image, component, capacity, degree, voltage, SN):
        with tf.device("/cpu:0"):
            # image = tf.nest.map_structure(tf.stop_gradient, tf.map_fn(parse_image, image, dtype=tf.float32))
            # degree = tf.nest.map_structure(tf.stop_gradient, tf.map_fn(lambda x: tf.one_hot(self.table_degree.lookup(x), 4), degree, dtype=tf.float32))
            image = tf.map_fn(parse_image, image, dtype=tf.float32, back_prop=False)
            # degree = tf.map_fn(self.table_degree.lookup, degree, dtype=tf.int64, back_prop=False)

        features = {
            "image": image,
            # "degree": degree,
        }
        logits = self.model(features)
        # predictions = {
        #     "output_node": logits
        # }
        # return predictions

        confidence = tf.math.reduce_max(input_tensor=logits, axis=-1, name="Confidence")
        pred_class = tf.math.argmax(input=logits, axis=-1)

        with tf.device("/cpu:0"):
            pred_class_str = tf.identity(self.table_label_str.lookup(pred_class))
            # pred_to_return = tf.identity(self.bool_to_pred_class.lookup(tf.cast(tf.math.equal(pred_class_str, degree), dtype=tf.int64)))

        predictions = {
            # "image": tf.identity(image[0][0][0][0]),
            "pred_class": pred_class_str,
            "confidence": confidence,
        }

        return predictions

def main():
    global h5_path, pb_dir, img_size, SAVED_PB_PATH
    h5_path = os.environ["h5_path"]
    pb_dir = os.environ["pb_dir"]
    img_size = os.environ["img_size"]
    version_name = os.environ["version_name"]
    model_from = os.environ["model_from"]
    label_num = os.environ["label_num"]
    SAVED_PB_PATH = pb_dir + version_name
    model = model_structure(model_from, (int(img_size),int(img_size),3), int(label_num))
    model.load_weights(h5_path)
    # model = tf.keras.models.load_model(h5_path)
    wrap = WrapModel(model)
    tf.saved_model.save(wrap, SAVED_PB_PATH)

if __name__ == "__main__":
    try: 
        main()
        print(f"H5: {h5_path} export to PB: {SAVED_PB_PATH}")
    except Exception as e:
        print(e)
