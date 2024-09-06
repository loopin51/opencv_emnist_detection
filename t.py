import tensorflow as tf
from tensorflow.python.client import device_lib
print(tf.__version__)
device_lib.list_local_devices()