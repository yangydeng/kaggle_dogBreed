'''
@Date  : 2017-11-16 14:11
@Author: yangyang Deng
@Email : yangydeng@163.com
'''

import tensorflow as tf
from tensorflow.python.framework import graph_util
import time
from Parameters import *


def save_model(sess, validation_loss):
    graph_def = tf.get_default_graph().as_graph_def()
    output_graph_def = graph_util.convert_variables_to_constants(sess, graph_def,
                                                                 ['BottleneckInputPlaceholder', 'final_tensor'])
    model_path_name = MODEL_SAVE_PATH+str(time.strftime('%Y-%m-%d_%H:%M',
                            time.localtime(time.time())))+'_'+str(validation_loss)+'_model.pb'

    with tf.gfile.GFile(model_path_name, 'wb') as f:
        f.write(output_graph_def.SerializeToString())

    print('model (' + model_path_name + ') save done!')