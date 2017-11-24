'''
@Date  : 2017-11-22 14:49
@Author: yangyang Deng
@Email : yangydeng@163.com
@Describe:
    修改图片路径，可创建train/train_supplement的TFRecord。
    按照一个batch50的大小，并行的处理得到每张图片的bottleneck，并将其存入TFRecord之中。
'''

import os
from algo.get_resnet_bottleneck import get_bottleneck
import tensorflow as tf
import pandas as pd
from tools.model_classes import Example_template
from Parameters import *


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


# 所有图片的文件名
train_image_names = os.listdir(TRAIN_IMAGE_DIR)
# 所有图片的总量
num_images = len(train_image_names)
# 存放所有图片的bottle_neck
bottle_necks_list = []
# 用于记录进行到了第几个batch
batch_count = 0
# 退出循环的标志
should_break = False
while True:
    # 每50个文件并行计算bottle_neck，提升速度
    batch_start = BOTTLE_NECK_BATCH_SIZE * batch_count
    batch_end = BOTTLE_NECK_BATCH_SIZE * (batch_count + 1)
    if batch_end>=num_images:
        batch_end = num_images
        should_break = True
    if batch_end==batch_start:
        break
    # 计数器
    count = 0
    image_path_list = []
    for train_image_name in train_image_names[batch_start:batch_end]:
        image_path_list.append(TRAIN_IMAGE_DIR + train_image_name)
        if count>=BOTTLE_NECK_BATCH_SIZE:
            break
        count += 1

    bottle_necks, _ = get_bottleneck(image_path_list)
    bottle_necks_list.extend(bottle_necks)
    batch_count+=1
    if batch_count%1==0:
        print('step bottle neck: ' + str(batch_count * BOTTLE_NECK_BATCH_SIZE))
    if should_break:
        break

labels = pd.read_csv(LABEL_PATH)
writer = tf.python_io.TFRecordWriter(TRAIN_OUTPUT_TFRECORD_PATH)

with tf.Session() as sess:
    for index in range(num_images):
        file_name = train_image_names[index]
        label = labels[labels.id==train_image_names[index].replace('.jpg','')].breed.tolist()[0]
        bottle_neck = ','.join(str(x) for x in bottle_necks_list[index])
        example = tf.train.Example(features=tf.train.Features(feature={
            Example_template.file_name: _bytes_feature(bytes(file_name, encoding="utf8")),
            Example_template.label: _bytes_feature(bytes(label, encoding ="utf8")),
            Example_template.bottle_neck: _bytes_feature(bytes(bottle_neck, encoding='utf8')),
        }))
        writer.write(example.SerializeToString())

writer.close()
print('file ' + TRAIN_OUTPUT_TFRECORD_PATH + ' has saved!')