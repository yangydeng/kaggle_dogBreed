Kaggle比赛中dog-breed-identification项目的源代码
比赛地址：https://www.kaggle.com/c/dog-breed-identification
本项目主要使用迁移学习的方法，采用的算法是 Inception-ResNet-v2。
目前score：0.22111, 排名：60+

一 使用流程：
    1. 使用/create_TFRecord目录下的文件计算bottleneck（瓶颈层）, 将bottle_neck存入TFRecord之后, 为后续使用节约大量时间;
    2. create_TFRecord目录下的文件会调用/algo/get_resnet_bottleneck.py来计算bottle_neck;
    3. 运行/app.py开始计算，运行中会同时进行预测并保存结果;
    4. Parameters中保存了模型所需要的所有参数, 方便调参。


二 文件功能说明：
    0. /app.py
        主程序

    1. /algo/get_resnet_bottleneck.py:
        调用ckpt文件计算bottle neck
       /algo/train_FC.py:
        以bottle_neck作为特征输入, 训练的FC层

    2. /create_TFRecord/creat_TFRecord_train.py
        计算train数据集的bottle_neck, 保存为TFRecord格式
       /create_TFRecord/create_TFRecord_test.py
        计算test数据集的bottle_neck, 保存为TFRecord格式

    3. /save_files/save_model.py
        保存训练好的FC模型
       /save_files/save_result.py
        保存训练好的预测结果, 可用于提交

    4. /TFslim
        下面的两个文件来自TF的github, 具体见/TFslim/readme

    5. /tools
        各种工具函数

    6. /Parameters
        参数集合