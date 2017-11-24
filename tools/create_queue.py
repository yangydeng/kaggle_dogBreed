'''
@Date  : 2017-11-23 19:27
@Author: yangyang Deng
@Email : yangydeng@163.com
@Describe: 
    
'''

import tensorflow as tf
from Parameters import *

def create_queue():
    reader = tf.TFRecordReader()
    # shuffle 在每个epoch当中，都会打乱文件的输入顺序
    train_filename_queue = tf.train.string_input_producer([TRAIN_TFrecord_path], shuffle=True, seed=1)
    _, train_serialized_example = reader.read(train_filename_queue)

    test_filename_queue = tf.train.string_input_producer([TEST_TFrecord_path], shuffle=False)
    _, test_serialized_example = reader.read(test_filename_queue)
    return train_serialized_example, test_serialized_example
