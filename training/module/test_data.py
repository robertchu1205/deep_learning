import module.tfrecord as tfrecord
import module.data as data
import glob
import tensorflow as tf
import numpy as np
# import matplotlib.pyplot as plt

TARGET_SHAPE = (96, 96, 3)
BATCH_SIZE = 1024
# filepath_with_stars = '/data/Phase2_11_01/*/AluCapacitor/OK/*/*.png' # new_filename -> False, inverse_bool=False
filepath_with_stars = '/data/Phase2_11_01/*/AluCapacitor/NG-InversePolarity/*.png' # new_filename -> False, inverse_bool=True 
# filepath_with_stars = '/data/alu-m2-img/Alu-M2-08231119/OK/*/*.png' # label from folder
# filepath_with_stars = '/data/alu-m2-img/Alu-M2-11201206/OK/*/*.png' # label from folder
# filepath_with_stars = '/data/alu-m2-img/stylized_alu_m2_08231119_img/*/*.png' # label from folder
# filepath_with_stars = '/data/alu-m2-img/dcgan_alu_m2_ng_img/*/*.png' # label from folder
# filepath_with_stars = '/data/Phase2_NewFN/*.png' # new_filename -> True, no need to define inverse_bool

# tfrecord_filepath = '/data/tfrecord/alu-m2-label/'+filepath_with_stars.split('/')[2]+'-Inverse.tfrecord'
# tfrecord_filepath = '/data/tfrecord/alu-m2/Phase2_11_01-Inverse.tfrecord'
split_ratio = 0.2

### tfrecord.py
# print dict of serialize example
# serialized_example = tfrecord.serialize_example(test_file, True)
# example_proto = tf.train.Example.FromString(serialized_example)
# print(example_proto)

# Write tfrecord

# tfrecord.write_tfrecord_file_from_filename(glob.glob(filepath_with_stars), tfrecord_filepath, False, True)
# tfrecord.write_tfrecord_file_from_folder(glob.glob(filepath_with_stars), tfrecord_filepath, False)

# Read tfrecord

# parsed_image_dataset = tfrecord.read_tfrecord_file(tfrecord_filepath)
# for image_features in parsed_image_dataset.take(2):
#     print(image_features)
    # plt.figure()
    # # plt.subplot(211)
    # plt.imshow(image_features['image'].numpy())
    # plt.title(image_features['file_path'].numpy())
    # plt.show()

# tfrecord to ready

# train_ds_1, valid_ds_1, train_step, valid_step = tfrecord.from_tfrecord_to_splitted_dataset(
#     tfrecord_filepath, TARGET_SHAPE, split_ratio, BATCH_SIZE)

# print(train_step)

# for image, label in train_ds_1.take(train_step):
#     # show_batch(image.numpy(),label.numpy())
#     # image, label is tf.Tensor
#     print("Image: ", image.numpy().shape)
#     print("Label: ", label.numpy())

# print(valid_step)

# for image, label in valid_ds_1.take(valid_step):
#     # show_batch(image.numpy(),label.numpy())
#     # image, label is tf.Tensor
#     print("Image: ", image.numpy().shape)
#     print("Label: ", label.numpy())

### data.py

# train_ds_1, valid_ds_1, train_step, valid_step = data.label_from_folder_to_splitted_dataset(
#     filepath_with_stars, TARGET_SHAPE, split_ratio, BATCH_SIZE
# )

# train_ds_1, valid_ds_1, train_step, valid_step = data.from_filepath_to_splitted_dataset(
#     filepath_with_stars, TARGET_SHAPE, False, split_ratio, BATCH_SIZE, True)

# print(train_step)

# for image, label in train_ds_1.take(train_step):
#     # show_batch(image.numpy(),label.numpy())
#     # image, label is tf.Tensor
#     print("Image: ", image.numpy().shape)
#     print("Label: ", label.numpy())

# print(valid_step)

# for image, label in valid_ds_1.take(valid_step):
#     # show_batch(image.numpy(),label.numpy())
#     # image, label is tf.Tensor
#     print("Image: ", image.numpy().shape)
#     print("Label: ", label.numpy())

# mix_ds = tf.data.experimental.sample_from_datasets([train_ds_1, valid_ds_1])

### test training.py dataset

test_img_tfrecord_list = [
    '/data/tfrecord/alu-m2/Alu-M2-08231119.tfrecord',
    '/data/tfrecord/alu-m2/Alu-M2-11201206.tfrecord']
(test_ds, test_step) = tfrecord.test_dataset_from_tfrecord(test_img_tfrecord_list, TARGET_SHAPE, 512)
ng_test_ds = test_ds.filter(lambda image, label: 
                    tf.keras.backend.eval(tf.math.argmax(input = label))==1 or 
                    tf.keras.backend.eval(tf.math.argmax(input = label))==3)

print(len(list(ng_test_ds)))
# count = 0
# for image, label in test_ds.take(ng_test_ds):
#     for l in label:
#         if np.argmax(l) == 3:
#             count+=1
#         elif np.argmax(l) == 1 :
#             count+=1
#             # print(np.argmax(l))
#     # print("@@@")
#     # print("Image: ", image.numpy().shape)
#     # print("Label: ", np.argmax(label.numpy()))
# print(count)