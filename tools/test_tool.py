'''
@Date  : 2017-11-24 10:41
@Author: yangyang Deng
@Email : yangydeng@163.com
@Describe: 
    
'''

from tools.get_next_batch import get_train_tensor
import tensorflow as tf
from tools.get_next_batch import drop_validate
import pandas as pd
from tools.model_classes import Example_template

def test1():
    with tf.Session() as sess:
        file_name_batch, label_batch, bottle_neck_batch = get_train_tensor(sess)
        for i in range(10**10):
            file_names, labels, bottle_necks = sess.run([file_name_batch, label_batch, bottle_neck_batch])
            print(i)



