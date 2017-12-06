'''
@Date  : 2017-11-17 20:40
@Author: yangyang Deng
@Email : yangydeng@163.com
'''

import tensorflow as tf
import matplotlib.pyplot as plt
from extend_images.picTransfer import preprocess_for_train


OUTPUT_RECORD_PATH = '/home/hiptonese/project/kaggle/dogBreedDataSet/TFRecordFiles/train.tfrecords'
TRAIN_SUPPLEMENT_PATH = '/home/hiptonese/project/kaggle/dogBreedDataSet/train_supplement/'
reader = tf.TFRecordReader()
TRAIN_NUM = 10222

# files = tf.train.match_filenames_once(OUTPUT_RECORD_PATH)
filename_queue = tf.train.string_input_producer([OUTPUT_RECORD_PATH], shuffle=False)

_, serialized_example = reader.read(filename_queue)

features = tf.parse_single_example(
    serialized_example,
    features={
        'image_raw': tf.FixedLenFeature([], tf.string),
        'label': tf.FixedLenFeature([], tf.string),
        'file_name': tf.FixedLenFeature([], tf.string),
    })

images_decode = tf.image.decode_jpeg(features['image_raw'])
labels = tf.cast(features['label'], tf.string)
file_names = tf.cast(features['file_name'], tf.string)

image_transfer = preprocess_for_train(images_decode)
image_convert = tf.image.convert_image_dtype(image_transfer, dtype=tf.uint8)
image_encode = tf.image.encode_jpeg(image_convert)

with tf.Session() as sess:
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    for i in range(TRAIN_NUM*5):
        # ！ 这里注意，不能在迭代中进行增加节点的操作，例如 tf.cast, tf.convert等等，否则计算机会因为节点数量过多而溢出。
        # !  这里，返回的结果应当用同一个sess.run()得到，不要在同一个循环中两次调用sess.run()，否则在多线程中可能会导致混乱。
        image, label, file_name, image_encode_eval = sess.run([images_decode, labels, file_names, image_encode])
        file_name = str(file_name, encoding='utf8').replace('.jpg', '')

        # 转换后的展示图片
        # plt.imshow(image_transfer.eval())
        # plt.title(file_name)
        # plt.show()

        with tf.gfile.GFile(TRAIN_SUPPLEMENT_PATH+file_name+'_'+str(i)+'.jpg', 'wb') as f:
            f.write(image_encode_eval)
        if i%100==0:
            print('step %d, pic: %s' % (i,file_name))

    coord.request_stop()
    coord.join(threads)


