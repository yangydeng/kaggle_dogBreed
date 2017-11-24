'''
@Date  : 2017-11-22 19:07
@Author: yangyang Deng
@Email : yangydeng@163.com
@Describe: 
    从TFRecord中取出bottle neck, 组成batch
'''

import tensorflow as tf
from tools.model_classes import Example_template, Usage
import pandas as pd
from Parameters import *


def get_next_tensor(sess, usage, validate_batch):
    if usage==Usage.validate:
        batch_data = validate_batch
        return batch_data
    elif usage==Usage.test:
        return get_test_tensor(sess)
    else:
        return get_train_tensor(sess)


def get_train_tensor(sess):
    reader = tf.TFRecordReader()
    # shuffle 在每个epoch当中，都会打乱文件的输入顺序
    filename_queue = tf.train.string_input_producer([TRAIN_TFrecord_path, TRAIN_Supplement_TFrecord_path], shuffle=True, seed=1)
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

    min_after_dequeue = BATCH_SIZE
    batch_size = BATCH_SIZE
    capacity = min_after_dequeue + 3*batch_size

    file_name_batch, label_batch, bottle_neck_batch = tf.train.shuffle_batch(
        [file_name, label, bottle_neck], batch_size=batch_size, capacity=capacity,
        min_after_dequeue=min_after_dequeue, num_threads=2
    )
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    return file_name_batch, label_batch, bottle_neck_batch


def get_test_tensor(sess):
    reader = tf.TFRecordReader()
    filename_queue = tf.train.string_input_producer([TEST_TFrecord_path], shuffle=False)

    _, serialized_example = reader.read(filename_queue)

    # test 数据只有file_name和bottle_neck, 没有标签。
    features = tf.parse_single_example(
        serialized_example,
        features={
            Example_template.file_name: tf.FixedLenFeature([], tf.string),
            Example_template.bottle_neck: tf.FixedLenFeature([], tf.string),
        })

    file_name = features[Example_template.file_name]
    bottle_neck = features[Example_template.bottle_neck]

    batch_size = NUM_TEST
    capacity = NUM_TEST*2

    file_name_batch, bottle_neck_batch = tf.train.batch(
        [file_name, bottle_neck], batch_size=batch_size, capacity=capacity,
    )
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    return file_name_batch, bottle_neck_batch


def test_make_dataframe(file_names, bottle_necks):
    file_names = [str(f, 'utf-8').replace('.jpg','') for f in file_names]
    bottle_necks = [[float(x) for x in str(b, 'utf-8').split(',')] for b in bottle_necks]
    batch_data = pd.DataFrame(
            columns=[Example_template.file_name, Example_template.bottle_neck])
    batch_data.file_name = file_names
    batch_data.bottle_neck = bottle_necks
    return batch_data


def train_make_dataframe(file_names, labels, bottle_necks):
    file_names = [str(f, 'utf-8').replace('.jpg','') for f in file_names]
    labels = [str(l, 'utf-8') for l in labels]
    bottle_necks = [[float(x) for x in str(b, 'utf-8').split(',')] for b in bottle_necks]
    batch_data = pd.DataFrame(
            columns=[Example_template.file_name, Example_template.label, Example_template.bottle_neck])
    batch_data.file_name = file_names
    batch_data.label = labels
    batch_data.bottle_neck = bottle_necks
    return batch_data


def drop_validate(df, validate_file_name):
    flag = df[Example_template.file_name].isin(validate_file_name)
    diff_flag = [not f for f in flag]
    res = df[diff_flag]
    res.index = [i for i in range(len(res))]
    return res

