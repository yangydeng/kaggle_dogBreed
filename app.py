'''
@Date  : 2017-11-23 12:57
@Author: yangyang Deng
@Email : yangydeng@163.com
@Describe: 
    训练的主函数入口。
'''

from tools.get_validation import get_validation
import tensorflow as tf
from algo.train_FC import train
from save_files.save_model import save_model
from save_files.save_result import save_result


def main(_):
    validate_batch, validate_file_name = get_validation()
    with tf.Session() as sess:
        test_result, test_image_ids, validation_cross_entropy_loss = train(sess, validate_batch, validate_file_name)
        save_result(test_result, test_image_ids, validation_cross_entropy_loss)
        save_model(sess, validation_cross_entropy_loss)


if __name__ == '__main__':
    tf.app.run()