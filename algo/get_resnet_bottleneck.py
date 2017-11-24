'''
@Date  : 2017-11-22 14:53
@Author: yangyang Deng
@Email : yangydeng@163.com
@Describe:
    通过传入的文件地址，计算出该文件对应的resnet bottleneck.
'''

import os
import tensorflow as tf
# from models.research.slim.nets import inception_resnet_v2
# from models.research.slim.preprocessing import inception_preprocessing
from TFslim import inception_resnet_v2
from TFslim import inception_preprocessing
from Parameters import *


def get_bottleneck(image_path_list, image_size = 299, is_training = False):
    slim = tf.contrib.slim
    with tf.Graph().as_default():
        with slim.arg_scope(inception_resnet_v2.inception_resnet_v2_arg_scope()):
            processed_images = []
            raw_images = []
            for image_path in image_path_list:
                image_raw = tf.gfile.FastGFile(image_path, 'rb').read()
                image_decode = tf.image.decode_jpeg(image_raw, channels=3)
                processed_image = inception_preprocessing.preprocess_image(image_decode, image_size, image_size, is_training=is_training)
                processed_images.append(processed_image)
                raw_images.append(image_raw)
            
            # if set num_classes=None, the line below will output the features we need to use migration learing!
            final_point, endpoints = inception_resnet_v2.inception_resnet_v2(processed_images, num_classes=None,
                                                                        is_training=False)
            final_point_flat = slim.flatten(final_point)
            init_fn = slim.assign_from_checkpoint_fn(
                os.path.join(CKPT_DIR, CKPT_FILENAME), slim.get_model_variables('InceptionResnetV2'))

            with tf.Session() as sess:
                init_fn(sess)
                final_point_eval = sess.run(final_point_flat)
                return final_point_eval,raw_images
