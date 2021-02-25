import time
import glob
import os
import tensorflow as tf 
import glob
# import argparse
from IPython import display

BATCH_SIZE = 256
# define training loop
MAX_DIM = 96
EPOCHS = 4000
noise_dim = 100
num_images_to_generate = 10
train_dataset_dir = ['/data/Alu-M2-08231119/OK/90/*.png', '/data/Alu-M2-11201206/OK/90/*.png']
BUFFER_SIZE = len(sorted(glob.glob(train_dataset_dir[0]))) + len(sorted(glob.glob(train_dataset_dir[1])))
print("BUFFER_SIZE:"+str(BUFFER_SIZE))
target_save_dir = '/data/dcgan_alu_m2_ng_img/90/'
if not os.path.exists(target_save_dir):
    os.makedirs(target_save_dir)
checkpoint_dir = '/tf/robertnb/dcgan_training_checkpoints/90'
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
prev_checkpoint_dir = '/tf/robertnb/dcgan_training_checkpoints/90'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")

# parser = argparse.ArgumentParser(description="max_dim of resized images, batch_size")
# parser.add_argument("-m", "--max_dim", dest="MAX_DIM", help="target shape of resized images")
# parser.add_argument("-b", "--batch_size", dest="BATCH_SIZE", help="batch_size for style transfer")
# args = parser.parse_args()
# MAX_DIM = int(args.MAX_DIM)
# BATCH_SIZE = int(args.BATCH_SIZE)

# This method returns a helper function to compute cross entropy loss
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def to_its_ds(root, cache=False):
    list_ds = tf.data.Dataset.list_files(root)
    image_ds = list_ds.map(process_path, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    image_ds = prepare_for_training(image_ds, cache=cache, shuffle_buffer_size=BUFFER_SIZE)
    return image_ds

def process_path(file_path):
    img = tf.io.read_file(file_path)
    img = decode_img(img) # input tensor is numpy array
    return img

def decode_img(img):
    img = tf.io.decode_png(img, channels=3)
#     img = tf.dtypes.as_dtype(tf.float32)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = (img - 0.5) / 0.5 # Normalize the images to [-1, 1]
    img = tf.image.resize(img, [MAX_DIM, MAX_DIM])
    return img

# For well shuffled, batched which gonna to be available
def prepare_for_training(ds, cache=False, shuffle_buffer_size=BUFFER_SIZE):
    if cache:
        if isinstance(cache, str):
            ds = ds.cache(cache)
        else:
            ds = ds.cache()
    ds = ds.shuffle(buffer_size=BUFFER_SIZE)
    # Repeat forever
    # ds = ds.repeat()
    ds = ds.batch(BATCH_SIZE)
    # `prefetch` lets the dataset fetch batches in the background while the model is training.
    ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return ds

ly = tf.keras.layers
def make_generator_model():
    model = tf.keras.Sequential()
    model.add(ly.Dense(12*12*256, use_bias=False, input_shape=(noise_dim,)))
    model.add(ly.BatchNormalization())
    model.add(ly.LeakyReLU())

    model.add(ly.Reshape((12, 12, 256)))
    assert model.output_shape == (None, 12, 12, 256) # Note: None is the batch size

    model.add(ly.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    assert model.output_shape == (None, 12, 12, 128)
    model.add(ly.BatchNormalization())
    model.add(ly.LeakyReLU())

    model.add(ly.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 24, 24, 64)
    model.add(ly.BatchNormalization())
    model.add(ly.LeakyReLU())
    
    model.add(ly.Conv2DTranspose(32, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 48, 48, 32)
    model.add(ly.BatchNormalization())
    model.add(ly.LeakyReLU())

    model.add(ly.Conv2DTranspose(3, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    assert model.output_shape == (None, MAX_DIM, MAX_DIM, 3)

    return model

def make_discriminator_model():
    model = tf.keras.Sequential()
    model.add(ly.Conv2D(32, (5, 5), strides=(2, 2), padding='same',
                                     input_shape=[MAX_DIM, MAX_DIM, 3]))
    model.add(ly.LeakyReLU())
    model.add(ly.Dropout(0.3))
    
    model.add(ly.Conv2D(64, (5, 5), strides=(2, 2), padding='same'))
    model.add(ly.LeakyReLU())
    model.add(ly.Dropout(0.3))

    model.add(ly.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(ly.LeakyReLU())
    model.add(ly.Dropout(0.3))

    model.add(ly.Flatten())
    model.add(ly.Dense(1))

    return model

# real images to an array of 1s, and the discriminator's predictions on fake (generated) images to an array of 0s
def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

# Notice the use of `tf.function`
# This annotation causes the function to be "compiled".
@tf.function
def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
      generated_images = generator(noise, training=True)

      real_output = discriminator(images, training=True)
      fake_output = discriminator(generated_images, training=True)

      gen_loss = generator_loss(fake_output)
      disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

def train_and_save_final_epoch_image(dataset, epochs):
    if epochs is not 0:
        for epoch in range(epochs):
            start = time.time()
            for image_batch in dataset:
                train_step(image_batch)
            # display.clear_output(wait=True)
            predictions = generator(seed, training=False)
            for i in range(predictions.shape[0]):
                print ('Generated image index "e{}-{}" is saved'.format(epoch, i))
                convert_and_save(predictions[i, :, :, :] * 0.5 + 0.5, i, epoch)

            # Save the model every 500 epochs
            # if (epoch + 1) % 500 == 0:
            #     checkpoint.save(file_prefix = checkpoint_prefix)
            print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))
    else:
        if os.environ["HAVE_CP"] == 'False':
            print("No restored checkpoints so breaks, otherwise, it'll be all dark!")
            return
    # Convert and save images after the final epoch
    # predictions = generator(seed, training=False)
    # for i in range(predictions.shape[0]):
    #     print ('Generated image index "e{}-{}" is saved'.format(epochs, i))
    #     convert_and_save(predictions[i, :, :, :] * 0.5 + 0.5, i, epochs)

def convert_and_save(image, index, epoch):
    file_target = target_save_dir+'d_c_gan_e'+str(epoch)+'_'+str(index)+'.png'
    img = tf.image.convert_image_dtype(image, tf.uint8)
#     img = tf.image.convert_image_dtype(predictions[0, :, :, :] * 0.5 + 0.5, tf.uint8)
    encoded_img = tf.image.encode_png(img)
    tf.io.write_file(
                        filename= file_target,
                        contents= encoded_img,
                        name=None
                    )

if __name__ == "__main__":
    dataset = []
    for d in train_dataset_dir:
        dataset.append(to_its_ds(d))
    train_dataset_ds = tf.data.experimental.sample_from_datasets(dataset)
    generator = make_generator_model()
    seed = tf.random.normal([num_images_to_generate, noise_dim])
    discriminator = make_discriminator_model()

    generator_optimizer = tf.keras.optimizers.Adam(1e-4)
    discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)
    checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)
    try:
        checkpoint.restore(tf.train.latest_checkpoint(prev_checkpoint_dir))
        os.environ["HAVE_CP"] = 'True'
    except:
        os.environ["HAVE_CP"] = 'False'
        print("no saved checkpoint")

    # mirrored_strategy = tf.distribute.MirroredStrategy(devices=["/gpu:0", "/gpu:1"], \
    #                     cross_device_ops=tf.distribute.NcclAllReduce())
    # with mirrored_strategy.scope():
    train_and_save_final_epoch_image(train_dataset_ds, EPOCHS)
