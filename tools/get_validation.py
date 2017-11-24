'''
@Date  : 2017-11-23 10:43
@Author: yangyang Deng
@Email : yangydeng@163.com
@Describe: 
    用于得到作为验证集的文件名/batch_data
'''

import tensorflow as tf
from tools.model_classes import Example_template
import pandas as pd
from sklearn.model_selection import train_test_split
from Parameters import *


def get_validation():
    reader = tf.TFRecordReader()
    # shuffle 在每个epoch当中，都会打乱文件的输入顺序
    filename_queue = tf.train.string_input_producer([TRAIN_TFrecord_path], shuffle=True, seed=1)

    _, serialized_example = reader.read(filename_queue)

    features = tf.parse_single_example(
        serialized_example,
        features={
            Example_template.file_name: tf.FixedLenFeature([], tf.string),
            Example_template.label: tf.FixedLenFeature([], tf.string),
            Example_template.bottle_neck: tf.FixedLenFeature([], tf.string),
        })

    file_name = features[Example_template.file_name]
    label = features[Example_template.label]
    bottle_neck = features[Example_template.bottle_neck]

    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        file_name_list = []
        label_list = []
        bottle_neck_list = []
        for i in range(NUM_TRAIN):
            file_name_eval, label_eval, bottle_neck_eval = sess.run([file_name, label, bottle_neck])

            file_name_eval = str(file_name_eval, 'utf-8').replace('.jpg','')
            label_eval = str(label_eval, 'utf-8')
            bottle_neck_eval = [float(str(x)) for x in str(bottle_neck_eval, 'utf-8').split(',')]

            file_name_list.append(file_name_eval)
            label_list.append(label_eval)
            bottle_neck_list.append(bottle_neck_eval)

        batch_data = pd.DataFrame(
            columns=[Example_template.file_name, Example_template.label, Example_template.bottle_neck])
        batch_data.file_name = file_name_list
        batch_data.label = label_list
        batch_data.bottle_neck = bottle_neck_list

        # 分离
        train_batch, validate_batch = train_test_split(batch_data,test_size=TEST_SIZE,random_state=1)
        train_batch = train_batch.reset_index(drop=True)
        validate_batch = validate_batch.reset_index(drop=True)
        validate_file_name = validate_batch[Example_template.file_name]
        coord.request_stop()
        coord.join(threads)
    return validate_batch, validate_file_name
