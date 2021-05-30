# This is a small code to check tensorflow version and whether tf gpu is installed 

import tensorflow as tf

if tf.test.gpu_device_name():
    print(tf.__version__)
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
else:
    print("Please install GPU version of TF")
