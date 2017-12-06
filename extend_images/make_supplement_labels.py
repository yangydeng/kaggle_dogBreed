'''
@Date  : 2017-11-19 12:31
@Author: yangyang Deng
@Email : yangydeng@163.com
'''

import pandas as pd
import os
import numpy as np

# OUTPUT_RECORD_PATH = '/home/hiptonese/project/kaggle/dogBreedDataSet/TFRecordFiles/train_supplement.tfrecords'
TRAIN_SUPPLEMENT_PATH = '/home/hiptonese/project/kaggle/dogBreedDataSet/train_supplement/'
LABEL_PATH = '/home/hiptonese/project/kaggle/dogBreedDataSet/labels/labels.csv'
SUPPLEMENT_LABEL_PATH = '/home/hiptonese/project/kaggle/dogBreedDataSet/labels/train_supplement_labels.csv'

fileNames = os.listdir(TRAIN_SUPPLEMENT_PATH)

ids_list = []
labels_list = []

count = 0
for fileName in fileNames:
    base_id = fileName.split('_')[0]
    id = fileName.replace('.jpg', '')
    labels = pd.read_csv(LABEL_PATH)
    label = labels[labels.id == base_id].breed.tolist()[0]
    ids_list.append(id)
    labels_list.append(label)
    count += 1
    if count%100 == 0:
        print(str(count)+' ' + id + ' ' + label)

data = np.array([ids_list, labels_list])
data = data.T

labels_supplement = pd.DataFrame(data,columns=['id', 'breed'])
labels_supplement.to_csv(SUPPLEMENT_LABEL_PATH,index=False)