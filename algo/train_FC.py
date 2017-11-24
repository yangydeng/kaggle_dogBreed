'''
@Date  : 2017-11-22 18:58
@Author: yangyang Deng
@Email : yangydeng@163.com
@Describe:
    训练全连层。
'''

import tensorflow as tf
from tools.get_next_batch import get_next_tensor
import pandas as pd
from tools.model_classes import Example_template
import numpy as np
from tools.model_classes import Usage
from save_files.save_result import save_result
from save_files.save_model import save_model
from tools.get_next_batch import train_make_dataframe
from tools.get_next_batch import test_make_dataframe
from tools.get_next_batch import drop_validate
from Parameters import *


sample = pd.read_csv(SAMPLE_OUTPUT_PATH)
breeds = list(sample.columns[1:])


def convert_labelstr2list(train_ground_truth_str):
    batch_size = len(train_ground_truth_str)
    ground_truth_list = []
    for i in range(batch_size):
        label_id = breeds.index(train_ground_truth_str[i])
        ground_truth = np.zeros(LABEL_NUMBERS, dtype=np.float32)
        ground_truth[label_id] = 1.
        ground_truth_list.append(ground_truth)
    ground_truth_list = np.array(ground_truth_list)
    return ground_truth_list


def convert_2D(train_bottlenecks):
    bottlenecks = []
    for item in train_bottlenecks:
        bottlenecks.append(item)
    return bottlenecks


def FC1():
    bottleneck_input = tf.placeholder(tf.float32, [None, BOTTLENECK_TENSOR_SIZE], name='bottle_neck_input_placeholder')
    groud_truth_input = tf.placeholder(tf.float32, [None, LABEL_NUMBERS], name='ground_truth')

    weights = tf.Variable(tf.truncated_normal([BOTTLENECK_TENSOR_SIZE, LABEL_NUMBERS], stddev=0.001))
    biases = tf.Variable(tf.zeros([LABEL_NUMBERS]))
    logits = tf.matmul(bottleneck_input, weights) + biases
    final_tensor = tf.nn.softmax(logits, name='final_tensor')
    normalize_weights([weights], reg=REGULARIZER)
    return bottleneck_input, groud_truth_input, logits, final_tensor


def train(sess, validate_batch, validate_file_name):
    bottleneck_input, groud_truth_input, logits, final_tensor = FC1()
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=groud_truth_input))
    loss = cross_entropy + tf.add_n(tf.get_collection('losses'))
    train_step = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)
    with tf.name_scope('evalution'):
        correct_prediction = tf.equal(tf.argmax(final_tensor, 1), tf.argmax(groud_truth_input, 1))
        evalution_step = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    train_file_name_tensor, train_label_tensor, train_bottle_neck_tensor = get_next_tensor(sess, Usage.train, validate_batch)
    test_file_name_tensor, test_bottle_neck_tensor = get_next_tensor(sess, Usage.test, validate_batch)
    init = tf.global_variables_initializer()
    sess.run(init)
    for i in range(STEPS):
        file_names, labels, bottle_necks = sess.run([train_file_name_tensor, train_label_tensor, train_bottle_neck_tensor])
        train_batch = train_make_dataframe(file_names, labels, bottle_necks)
        # 删除用作 验证集 的样本
        train_batch = drop_validate(train_batch, validate_file_name)
        train_bottlenecks, train_ground_truth_str = train_batch[Example_template.bottle_neck], train_batch[Example_template.label]
        train_bottlenecks = convert_2D(train_bottlenecks)
        train_ground_truth = convert_labelstr2list(train_ground_truth_str)

        sess.run(train_step, feed_dict={bottleneck_input: train_bottlenecks, groud_truth_input: train_ground_truth})
        if i % 100 == 0 or i + 1 == STEPS:
            cross_entropy_loss = sess.run(cross_entropy, feed_dict={bottleneck_input: train_bottlenecks,
                                                                    groud_truth_input: train_ground_truth})
            print('Step %d: Train cross entropy on random sampled %d examples = %.3f' % (i, BATCH_SIZE, cross_entropy_loss))

        # 每X步产出一次结果
        if i % STEPS_OUTPUT == 0:
            validate_batch_data = get_next_tensor(sess, Usage.validate, validate_batch)
            validate_bottlenecks, validate_ground_truth_str = validate_batch_data[Example_template.bottle_neck], validate_batch_data[Example_template.label]
            validate_bottlenecks = convert_2D(validate_bottlenecks)
            validate_ground_truth = convert_labelstr2list(validate_ground_truth_str)

            validation_cross_entropy_loss = sess.run(
                cross_entropy, feed_dict={bottleneck_input: validate_bottlenecks, groud_truth_input: validate_ground_truth})

            file_names, bottle_necks = sess.run([test_file_name_tensor, test_bottle_neck_tensor])
            test_batch = test_make_dataframe(file_names, bottle_necks)
            test_bottlenecks, test_image_ids = test_batch[Example_template.bottle_neck], test_batch[Example_template.file_name]
            test_bottlenecks = convert_2D(test_bottlenecks)
            test_image_ids = convert_2D(test_image_ids)
            test_result = sess.run(final_tensor, feed_dict={bottleneck_input: test_bottlenecks})
            save_result(test_result, test_image_ids, validation_cross_entropy_loss)

    validate_batch_data = get_next_tensor(sess, Usage.validate, validate_batch)
    validate_bottlenecks, validate_ground_truth_str = validate_batch_data[Example_template.bottle_neck], validate_batch_data[Example_template.label]
    validate_ground_truth = convert_labelstr2list(validate_ground_truth_str)
    validation_multiclass_loss = sess.run(evalution_step,
                                          feed_dict={bottleneck_input: validate_bottlenecks,
                                                     groud_truth_input: validate_ground_truth})
    validation_multiclass_loss = 1 - validation_multiclass_loss
    print('Final validation multiclass loss is %.5f%%' % validation_multiclass_loss)

    validation_cross_entropy_loss = sess.run(cross_entropy, feed_dict={bottleneck_input: validate_bottlenecks,
                                                                       groud_truth_input: validate_ground_truth})
    print('Final validation cross entropy is %.5f' % validation_cross_entropy_loss)

    file_names, bottle_necks = sess.run([test_file_name_tensor, test_bottle_neck_tensor])
    test_batch = test_make_dataframe(file_names, bottle_necks)
    test_bottlenecks, test_image_ids = test_batch[Example_template.bottle_neck], test_batch[Example_template.file_name]
    test_bottlenecks = convert_2D(test_bottlenecks)
    test_image_ids = convert_2D(test_image_ids)
    test_result = sess.run(final_tensor, feed_dict={bottleneck_input: test_bottlenecks})
    return test_result, test_image_ids, validation_cross_entropy_loss


def normalize_weights(list_weights, reg='l2'):
    if reg == 'l2':
        regularizer = tf.contrib.layers.l2_regularizer(REGULARAZTION_RATE)
        for weight in list_weights:
            tf.add_to_collection('losses', regularizer(weight))
    elif reg == 'l1':
        regularizer = tf.contrib.layers.l1_regularizer(REGULARAZTION_RATE)
        for weight in list_weights:
            tf.add_to_collection('losses', regularizer(weight))
    elif reg=='l1+l2':
        regularizer_l1 = tf.contrib.layers.l1_regularizer(REGULARAZTION_RATE)
        regularizer_l2 = tf.contrib.layers.l2_regularizer(REGULARAZTION_RATE)
        for weight in list_weights:
            tf.add_to_collection('losses', regularizer_l1(weight))
            tf.add_to_collection('losses', regularizer_l2(weight))


