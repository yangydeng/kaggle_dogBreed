'''
@Date  : 2017-11-23 13:00
@Author: yangyang Deng
@Email : yangydeng@163.com
@Describe: 
    存放了训练用到的各种参数。
'''

# 训练集的总数
NUM_TRAIN = 10222
# 训练集拓展集的总数（通过图片的转换得到拓展集）
NUM_TRAIN_SUPPLEMENT = 51110
# 测试集的总数
NUM_TEST =10357
# 验证集所占训练集的比例
TEST_SIZE = 0.02

# TRAIN_TFRecord文件位置
TRAIN_TFrecord_path = '/home/hiptonese/project/kaggle/dogBreedDataSet/TFRecordFiles/resnet/resnet_train.tfrecords'
# TRAIN_Supplement_TFRecord文件的存放位置
TRAIN_Supplement_TFrecord_path = '/home/hiptonese/project/kaggle/dogBreedDataSet/TFRecordFiles/resnet/resnet_train_supplement.tfrecords'
# TEST_TFRecord文件位置
TEST_TFrecord_path = '/home/hiptonese/project/kaggle/dogBreedDataSet/TFRecordFiles/resnet/resnet_test.tfrecords'
# 官方给的提交范例
SAMPLE_OUTPUT_PATH = '/home/hiptonese/project/kaggle/dogBreed/sample_submission.csv'
# 保存提交结果的路径
RESULT_SAVE_PATH = '/home/hiptonese/project/kaggle/dogBreedDataSet/results/'
# 训练后模型文件的保存路径
MODEL_SAVE_PATH = '/home/hiptonese/project/kaggle/dogBreedDataSet/model_save/'
# 用于训练的图片存放路径
TRAIN_IMAGE_DIR = '/home/hiptonese/project/kaggle/dogBreedDataSet/train_supplement/'
# 存放所有标签的表格
LABEL_PATH = '/home/hiptonese/project/kaggle/dogBreedDataSet/labels/all_labels.csv'
# 训练图片转化成TFRecord之后的存放路径
TRAIN_OUTPUT_TFRECORD_PATH = '/home/hiptonese/project/kaggle/dogBreedDataSet/TFRecordFiles/resnet/resnet_train.tfrecords'
# 每次计算batch数量的图片bottleneck, 100会导致死机
BOTTLE_NECK_BATCH_SIZE = 20
# TEST图片的存放位置
TEST_IMAGE_DIR = '/home/hiptonese/project/kaggle/dogBreedDataSet/test/'
# 将TEST的bottleneck放入TFRecord后的存放路径
TEST_OUTPUT_TFRECORD_PATH = '/home/hiptonese/project/kaggle/dogBreedDataSet/TFRecordFiles/restnet/resnet_test.tfrecords'
# ckpt文件存放的路径
CKPT_DIR = '/home/hiptonese/project/kaggle/dogBreedDataSet/ckpt_files/'
# ckpt文件的名称
CKPT_FILENAME = 'inception_resnet_v2_2016_08_30.ckpt'

# bottle_neck的数量
BOTTLENECK_TENSOR_SIZE = 1536
# 输出层的数量
LABEL_NUMBERS = 120
# 训练速率
LEARNING_RATE = 0.0001
# 训练迭代的总数
STEPS = (NUM_TRAIN+NUM_TRAIN_SUPPLEMENT)*3
# 每step_out步产出一次结果
STEPS_OUTPUT = 500
# 训练用的batch_size
BATCH_SIZE = 1000
# 正则化权重
REGULARAZTION_RATE = 0.0001
# 正则化的种类
regularizer = ['l1', 'l2', 'l1+l2']
# 正则化赋值
REGULARIZER = regularizer[2]


