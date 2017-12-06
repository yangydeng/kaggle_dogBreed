'''
@Date  : 2017-11-17 19:35
@Author: yangyang Deng
@Email : yangydeng@163.com
'''

import tensorflow as tf
import os
import pandas as pd

LABEL_PATH = '/home/hiptonese/project/kaggle/dogBreedDataSet/labels/labels.csv'
OUTPUT_RECORD_PATH = '/home/hiptonese/project/kaggle/dogBreedDataSet/TFRecordFiles/train.tfrecords'
train_dataset_dir = '/home/hiptonese/project/kaggle/dogBreedDataSet/train/'


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


# 读取图片原始数据。
fileNames = os.listdir(train_dataset_dir)
labels = pd.read_csv(LABEL_PATH)
writer = tf.python_io.TFRecordWriter(OUTPUT_RECORD_PATH)

count = 0
with tf.Session() as sess:
    for fileName in fileNames:
        image_raw = tf.gfile.FastGFile(train_dataset_dir + fileName, 'rb').read()
        label = labels[labels.id==fileName.replace('.jpg','')].breed.tolist()[0]
        example = tf.train.Example(features=tf.train.Features(feature={
            'image_raw': _bytes_feature(image_raw),
            'label': _bytes_feature(bytes(label, encoding = "utf8")),
            'file_name': _bytes_feature(bytes(fileName, encoding = "utf8"))
        }))
        writer.write(example.SerializeToString())
        if count%1000 == 0:
            print('step: %d, file name: %s' %(count, fileName))
        count+=1

writer.close()
print('file ' + OUTPUT_RECORD_PATH + ' has saved!')



