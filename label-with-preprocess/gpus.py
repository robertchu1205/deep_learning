import os
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
   try:
       for gpu in gpus:
           #print(gpu)
           tf.config.experimental.set_memory_growth(gpu, True)
       logical_gpus = tf.config.experimental.list_logical_devices('GPU')
       print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
   except RuntimeError as e:
       print(e)
