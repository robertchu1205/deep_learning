import os
import shutil
from absl import app
import datetime
import module.model as model
# import module.metadata as metadata
import sys
import sqlite3

import tensorflow as tf
from tensorboard.plugins.hparams import api as hp
import neural_structured_learning as nsl

def initialized_tf():
    # tf methods
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    print(physical_devices)
    # assert len(physical_devices) > 0, 'Not enough GPU hardware devices available'
    # if physical_devices:
    #     try:
    #         for gpu in physical_devices:
    #             tf.config.experimental.set_memory_growth(gpu, True)
    #     except RuntimeError as e:
    #         # Memory growth must be set before GPUs have been initialized
    #         print(e)
    if all_var_dict['MP_POLICY'] is True:
        policy = tf.keras.mixed_precision.experimental.Policy('mixed_float16')
        tf.keras.mixed_precision.experimental.set_policy(policy)
    else:
        policy = tf.keras.mixed_precision.experimental.Policy('float32')
        tf.keras.mixed_precision.experimental.set_policy(policy)

# hparams 
def define_hparams():
    global HP_conv_count, HP_first_filter, HP_kernel_size
    global HP_strides, HP_maxpool, HP_learning_rate, HP_optimizer
    global HP_augment, HP_adversarial, HP_adv_step_size
    HP_conv_count = hp.HParam('conv_count', hp.IntInterval(3, 4))
    HP_first_filter = hp.HParam('first_filter', hp.Discrete([8, 16, 32, 64])) # 256, 512, 1024: OOM
    HP_kernel_size = hp.HParam('kernel_size', hp.Discrete([2, 3, 5]))
    HP_strides = hp.HParam('strides', hp.Discrete([1, 2, 3]))
    HP_maxpool = hp.HParam('maxpool', hp.Discrete([True, False]))
    HP_learning_rate = hp.HParam('learning_rate', hp.IntInterval(-5, -3))
    # HP_augment = hp.HParam('augment', hp.Discrete([True, False]))
    HP_adversarial = hp.HParam('adversarial', hp.Discrete([True]))
    HP_adv_step_size = hp.HParam('adv_step_size', hp.RealInterval(1e-2, 2e-1))
    HP_optimizer = hp.HParam('optimizer', hp.Discrete(['adam', 'adagrad', 'RMSprop']))
    global HPARAMS_LIST
    HPARAMS_LIST = [ 
        HP_conv_count, HP_first_filter, HP_kernel_size, HP_strides, HP_maxpool, 
        HP_learning_rate, HP_optimizer, HP_adversarial, HP_adv_step_size]
    if all_var_dict['gan_ds_list'] != []:
        global HP_generative_ratio, HP_dcgan_ratio
        HP_generative_ratio = hp.HParam('generative_ratio', hp.IntInterval(0, 5))
        HP_dcgan_ratio = hp.HParam('dcgan_ratio', hp.RealInterval(0., 1.))
        HPARAMS_LIST.extend([HP_generative_ratio, HP_dcgan_ratio])

# functions
def prepare_data(hparams):
    # metadata.assign_global(all_var_dict)
    if all_var_dict['gan_ds_list'] != []:
        train_ds, valid_ds = all_var_dict['prepare_function'](
                                hparams[HP_generative_ratio], all_var_dict['gan_ds_list'], 
                                [1 - hparams[HP_dcgan_ratio], hparams[HP_dcgan_ratio]], 
                                augment=hparams[HP_augment])
    else:
        train_ds, valid_ds = all_var_dict['prepare_function']()
    train_valid_data = (train_ds, valid_ds)
    if all_var_dict['test_ds'] == []:
        test_data = ''
    else:
        test_data = all_var_dict['test_ds_to_eva'](all_var_dict['test_ds'])
    return train_valid_data, test_data

def model_fn(hparams, output_bias=None):
    # deal with inbalanced data
    if output_bias is not None:
        output_bias = tf.keras.initializers.Constant(output_bias)
    try:
        # loaded_model = model.BaseModel()
        # loaded_model = model.ConcatConv(hparams[HP_conv_count], hparams[HP_first_filter], 
        #                                 hparams[HP_kernel_size], hparams[HP_strides], 
        #                                 all_var_dict['target_shape'], all_var_dict['DEGREE_NUM'],
        #                                 all_var_dict['LABEL_NUM'], hparams[HP_maxpool])
        loaded_model = model.SomeConv(hparams[HP_conv_count], hparams[HP_first_filter], 
                                        hparams[HP_kernel_size], hparams[HP_strides], 
                                        all_var_dict['target_shape'], all_var_dict['LABEL_NUM'], 
                                        hparams[HP_maxpool])
    except Exception as e:
        print(f'Exception msg: {e}')
        print((f'{hparams[HP_conv_count]}, {hparams[HP_first_filter]}'
                f'{hparams[HP_kernel_size]}, {hparams[HP_strides]}, {hparams[HP_maxpool]}'))
        return False
    # Wrap the model with adversarial regularization.
    if hparams[HP_adversarial]:
        adv_config = nsl.configs.make_adv_reg_config(multiplier=2e-1, 
                                                        adv_step_size=hparams[HP_adv_step_size], 
                                                        adv_grad_norm = 'infinity')
        adv_loaded_model = nsl.keras.AdversarialRegularization(loaded_model, 
                                                        label_keys=['label'],
                                                        adv_config=adv_config)
    # print(loaded_model.summary())
    if hparams[HP_optimizer]== 'adam':
        optimizer=tf.keras.optimizers.Adam(learning_rate = 10 ** hparams[HP_learning_rate])
    elif hparams[HP_optimizer]== 'adagrad':
        optimizer=tf.keras.optimizers.Adagrad(learning_rate = 10 ** hparams[HP_learning_rate])
    elif hparams[HP_optimizer]== 'Adadelta':
        optimizer=tf.keras.optimizers.Adadelta(learning_rate = 10 ** hparams[HP_learning_rate])
    elif hparams[HP_optimizer]== 'SGD':
        optimizer=tf.keras.optimizers.SGD(learning_rate = 10 ** hparams[HP_learning_rate])
    adv_loaded_model.compile(optimizer=optimizer, 
                        # loss=tf.keras.losses.BinaryCrossentropy(), 
                        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),  
                        metrics=all_var_dict['METRICS'])
    return adv_loaded_model

def run(train_valid_data, test_data, hparams):
    (train_ds, valid_ds) = train_valid_data
    try:
        if all_var_dict['DISTRIBUTED'] != None:
            with all_var_dict['DISTRIBUTED'].scope():
                loaded_model = model_fn(hparams=hparams)
        else:
            loaded_model = model_fn(hparams=hparams)
        if loaded_model == False:
            return False
    except Exception as e:
        print(f'Exception msg: {e}')
        return False
    to_append_folder_name = ( f'cc-{str(hparams[HP_conv_count])}_'
                            f'ff-{str(hparams[HP_first_filter])}_'
                            f'ks-{str(hparams[HP_kernel_size])}_'
                            f's-{str(hparams[HP_strides])}_'
                            f'mp-{str(hparams[HP_maxpool])}_'
                            f'lr{str(hparams[HP_learning_rate])}_'
                            f'op-{str(hparams[HP_optimizer])}_'
                            # f'aug-{str(hparams[HP_augment])}_'
                            f'adv-{str(hparams[HP_adversarial])}' )
    tb_log_dir = os.path.join(all_var_dict['base_tb_dir'], to_append_folder_name)
    trained_h5_dir = f'''{all_var_dict['base_h5_dir']}{to_append_folder_name}/'''
    if not os.path.exists(trained_h5_dir):
        os.makedirs(trained_h5_dir)
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', 
            patience=5, 
            mode='min', 
            restore_best_weights=True,
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir = tb_log_dir,
            # histogram_freq=2,
            write_graph=False,
            # write_images=True,
            profile_batch=0,
        ),
        hp.KerasCallback(tb_log_dir, hparams),  # log hparams
        tf.keras.callbacks.ModelCheckpoint(
            filepath= trained_h5_dir+'ep_{epoch:03d}-vl_{val_loss:03f}-va_{val_acc:03f}.h5',
            save_best_only=True,
            monitor='val_loss',
            mode='min',
        ),
    ]
    training_history = loaded_model.fit(train_ds, 
                epochs=all_var_dict['EPOCH'], 
                validation_data=valid_ds, 
                steps_per_epoch=all_var_dict['train_step'], 
                # validation_steps=all_var_dict['valid_step'], 
                callbacks=callbacks,
                verbose=1 # 0 = silent, 1 = progress bar, 2 = one line per epoch
            )
    loaded_model.save(f'{trained_h5_dir}final.h5') # <=2.2
    # tf.keras.models.save_model(loaded_model, f'{trained_h5_dir}final', save_format='h5') # 2.3
    test_data_perform = ''
    if test_data != '':
        test_data_perform = loaded_model.evaluate(test_data)
        named_results = dict(zip(loaded_model.metrics_names, test_data_perform))
        with tf.summary.create_file_writer(f'{tb_log_dir}/test').as_default():
            hp.hparams(hparams)
            for key, value in named_results.items():
                tf.summary.scalar(f'test.{key}', value, step=1)
    return test_data_perform

def run_all(verbose=False):
    if all_var_dict['random_times'] == 0:
        print(f'Did not specify random_times')
    else: 
        for time in range(all_var_dict['random_times']):
            test_data_perform = False
            while test_data_perform == False:
                hparams = {h: h.domain.sample_uniform() for h in HPARAMS_LIST}
                if all_var_dict['gan_ds_list'] != []:
                    hparams[HP_generative_ratio] = float(int(hparams[HP_generative_ratio]) * 1e-1)
                if verbose:
                    print(f'''--- {time+1}/{all_var_dict['random_times']} random times ---''')
                    for key, value in hparams.items():
                        print(key, value)
                train_valid_data, test_data=prepare_data(hparams)
                test_data_perform = run(
                    train_valid_data=train_valid_data, 
                    test_data=test_data,  
                    hparams=hparams, 
                )
                if verbose:
                    if test_data_perform != False:
                        print(f'test_data_perform: {test_data_perform}')

def main(ALL_VAR_DICT):
    global all_var_dict
    all_var_dict = ALL_VAR_DICT
    initialized_tf()
    define_hparams()
    shutil.rmtree(all_var_dict['base_tb_dir'], ignore_errors=True)
    shutil.rmtree(all_var_dict['base_h5_dir'], ignore_errors=True)
    if all_var_dict['LOG_VERBOSE'] is True:
        os.makedirs(all_var_dict['base_h5_dir'])
        log_location = f'''{all_var_dict['base_h5_dir']}{all_var_dict['dir_basename']}.log'''
        print(f'Create log to {log_location}')
        stdout = sys.stdout
        sys.stdout = open(log_location, 'w')
    print(f'''Saving output to {all_var_dict['base_tb_dir']}.''')
    run_all(verbose=all_var_dict['RUN_ALL_VERBOSE'])
    print(f'''Done. Output saved to {all_var_dict['base_tb_dir']}.''')
    if all_var_dict['LOG_VERBOSE'] is True:
        sys.stdout = stdout
        sys.stdout.close()

if __name__ == "__main__":
    app.run(main)