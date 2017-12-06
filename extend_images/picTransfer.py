'''
@Date  : 2017-11-17 13:11
@Author: yangyang Deng
@Email : yangydeng@163.com
'''

import tensorflow as tf
import numpy as np

BRIGHTNESS_DELTA = 64./255. #32./255.
SATURATION = [0.5, 1.5]
HUE_DELTA = 0.05
CONTRAST = [0.5, 1.5]


# 将亮度颜色等调节得更加明显
def distort_color(image, color_ordering=0):
    if color_ordering == 0:
        image = tf.image.random_brightness(image, max_delta=BRIGHTNESS_DELTA)
        image = tf.image.random_saturation(image, lower=SATURATION[0], upper=SATURATION[1])
        image = tf.image.random_hue(image, max_delta=HUE_DELTA)
        image = tf.image.random_contrast(image, lower=CONTRAST[0], upper=CONTRAST[1])
    elif color_ordering == 1:
        image = tf.image.random_saturation(image, lower=SATURATION[0], upper=SATURATION[1])
        image = tf.image.random_brightness(image, max_delta=BRIGHTNESS_DELTA)
        image = tf.image.random_contrast(image, lower=CONTRAST[0], upper=CONTRAST[1])
        image = tf.image.random_hue(image, max_delta=HUE_DELTA)
    elif color_ordering ==2 :
        image = tf.image.random_hue(image, max_delta=HUE_DELTA)
        image = tf.image.random_contrast(image, lower=CONTRAST[0], upper=CONTRAST[1])
        image = tf.image.random_saturation(image, lower=SATURATION[0], upper=SATURATION[1])
        image = tf.image.random_brightness(image, max_delta=BRIGHTNESS_DELTA)
    else:
        image = tf.image.random_contrast(image, lower=CONTRAST[0], upper=CONTRAST[1])
        image = tf.image.random_hue(image, max_delta=HUE_DELTA)
        image = tf.image.random_saturation(image, lower=SATURATION[0], upper=SATURATION[1])
        image = tf.image.random_brightness(image, max_delta=BRIGHTNESS_DELTA)
    # 由于每个像素点的RGB都是[0,1]之间的，为了安全要处理
    return tf.clip_by_value(image, 0.0, 1.0)


def preprocess_for_train(image, bbox=None):
    # 查看是否存在标注框。
    if bbox is None:
        bbox = tf.constant([0.0, 0.0, 1.0, 1.0], dtype=tf.float32, shape=[1, 1, 4])
    # if image.dtype != tf.float32:
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)

    # 随机的截取图片中一个块。
    bbox_begin, bbox_size, _ = tf.image.sample_distorted_bounding_box(
        tf.shape(image), bounding_boxes=bbox, area_range=[0.4,0.9])
    distorted_image = tf.slice(image, bbox_begin, bbox_size)

    # 将随机截取的图片调整为神经网络输入层的大小。
    distorted_image = tf.image.random_flip_left_right(distorted_image)
    distorted_image = distort_color(distorted_image, np.random.randint(4))
    return tf.clip_by_value(distorted_image, 0.0, 1.0)


