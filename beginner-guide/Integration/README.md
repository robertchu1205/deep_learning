# Purpose 
###### Transform pretrained model to servable .pb file

# Instances
1. A source model to .pb
2. Multiple models to .pb

# Detail
The example code shows how to combine 2 models outcome to 'OK' or 'NG'. <br>
These two models have different inputs and outputs. Therefore, it needs to **_combine two outputs_** with some logical calculation. <br>
Besides, some variable are defined in **_environment variable and argparse_** for more convenience of altering variable while moving file.

# Attention
1. Tried converting keras model to estimator and to .pb with exported_saved_model - **_Failed_** <br> - Estimator model graph is **_still trainable_** when setting keras’s layers trainable = False
2. Must release **_GPU processes_** when create a docker container for tf serving by gpu
3. Some keras models will add activation **_“softmax”_** in the end, so adding another “softmax” is unnecessary
4. Serving input tensor must be **_the same of training’s_**, otherwise the prediction will be totally different
5. Tensorflow lookup table **_CAN NOT_** be put in serving_parse_fn, otherwise the placeholder will be in different graph, <br> even happen **_Invalid Argument_** ( must feed a value for placeholder tensor )


