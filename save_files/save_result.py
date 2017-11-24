'''
@Date  : 2017-11-16 13:55
@Author: yangyang Deng
@Email : yangydeng@163.com
'''

import numpy as np
import pandas as pd
import time
from Parameters import *


def save_result(test_result, test_image_ids, validation_loss):
    test_result = np.array(test_result)
    test_image_ids = np.array(test_image_ids).reshape((len(test_image_ids), 1))

    data = np.concatenate((test_image_ids, test_result), axis=1)

    columns = pd.read_csv(SAMPLE_OUTPUT_PATH).columns
    res = pd.DataFrame(data, columns=columns)
    # save result
    result_path_name = RESULT_SAVE_PATH+str(time.strftime('%Y-%m-%d_%H:%M',time.localtime(time.time())))+'_'+str(validation_loss)+'_res.csv'
    res.to_csv(result_path_name, index=False)

    print('Result (' + result_path_name + ') done!')
