import os, cv2
import numpy as np
import tensorflow as tf

LABEL_CLASS_LIST = ['000', '090', '180', '270', 'NG']

class TFModel():
    def __init__(self, classifier, img_size=(640, 640, 1)):
        self.Model = tf.keras.models.load_model(classifier)
        self.image_size = img_size
        
    # def ssim_loss(self, ytrue ,ypred):
    #     return(1 - tf.image.ssim(ytrue ,ypred, max_val=1))
    
    def prepare_img(self, img):
        img = tf.io.decode_jpeg(img, channels=1, dct_method='INTEGER_ACCURATE', try_recover_truncated=True)
        img = tf.cast(img, dtype=tf.dtypes.float32) / 255.0
        img = tf.image.resize_with_pad(img, self.image_size[1], self.image_size[0])
        return img

    def test_func(self, path):
        features = {
            'image': self.prepare_img(tf.io.read_file(path)),
        }
        return features

    def filter_ans(self, arr):
        return {'pred_class': LABEL_CLASS_LIST[np.argmax(arr)], 'confidence': float(np.max(arr))}
    
    def test_step(self, paths):
        ans = []
        test_ds = tf.data.Dataset.from_tensor_slices(tf.constant(paths))
        input_images = test_ds.map(self.test_func, tf.data.experimental.AUTOTUNE).batch(len(paths))
        predicted = self.Model.predict(input_images)
        for pre in predicted:
            ans.append(self.filter_ans(pre))
        return ans