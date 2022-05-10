# coding: UTF-8
import datetime
import logging
import os
import time
import torch
import numpy as np
from train_eval import train, init_network
from importlib import import_module
# import argparse
from utils import build_dataset, build_iterator, get_time_dif

# parser = argparse.ArgumentParser(description='Chinese Text Classification')
# parser.add_argument('--model', type=str, required=True, help='choose a model: Bert, ERNIE')
# args = parser.parse_args()

dataset = 'text'  # 数据集
model_name = 'bert'
nowTime = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
string_pattern = "{}_{}".format(model_name, nowTime)


if __name__ == '__main__':
    # dataset = 'THUCNews'  # 数据集

    x = import_module('models.' + model_name)
    config = x.Config(dataset)
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True  # 保证每次结果一样

    start_time = time.time()
    print("Loading data...")
    train_data, dev_data, test_data = build_dataset(config)

    train_iter = build_iterator(train_data, config)

    dev_iter = build_iterator(dev_data, config)

    test_iter = build_iterator(test_data, config)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)

    log_output = './output'



    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(filename)s %(levelname)s %(message)s',
                        datefmt='%a, %d %b %Y %H:%M:%S',
                        filename=os.path.join(log_output, '{}.log'.format(string_pattern)),
                        filemode='a')
    logging.info('{}.log\n\n'.format(string_pattern))

    # train
    model = x.Model(config).to(config.device)
    print(len(train_iter))
    train(config, model, train_iter, dev_iter, test_iter,string_pattern)

"""
Precision, Recall and F1-Score...
              precision    recall  f1-score   support
          好评     0.9563    0.9386    0.9473      2051
          中评     0.6607    0.6372    0.6487      1039
          差评     0.6955    0.7490    0.7213       976
    accuracy                         0.8160      4066
   macro avg     0.7708    0.7749    0.7724      4066
weighted avg     0.8182    0.8160    0.8168      4066
Confusion Matrix...
[[1925  107   19]
 [  76  662  301]
 [  12  233  731]]
Time usage: 0:00:06



Test Loss:  0.47,  Test Acc: 80.23%
Precision, Recall and F1-Score...
              precision    recall  f1-score   support
          好评     0.9494    0.9239    0.9365      2051
          中评     0.6452    0.5967    0.6200      1039
          差评     0.6736    0.7654    0.7165       976
    accuracy                         0.8023      4066
   macro avg     0.7560    0.7620    0.7577      4066
weighted avg     0.8054    0.8023    0.8028      4066
Confusion Matrix...
[[1895  128   28]
 [  85  620  334]
 [  16  213  747]]
Time usage: 0:00:03


Test Loss:  0.45,  Test Acc: 81.80%
Precision, Recall and F1-Score...
              precision    recall  f1-score   support
          好评     0.9556    0.9439    0.9497      2051
          中评     0.6834    0.6044    0.6415      1039
          差评     0.6798    0.7807    0.7268       976
    accuracy                         0.8180      4066
   macro avg     0.7729    0.7764    0.7726      4066
weighted avg     0.8198    0.8180    0.8174      4066
Confusion Matrix...
[[1936   91   24]
 [  76  628  335]
 [  14  200  762]]
Time usage: 0:00:06

"""

"""
No optimization for a long time, auto-stopping...
Test Loss:  0.17,  Test Acc: 94.42%
Precision, Recall and F1-Score...
               precision    recall  f1-score   support
      finance     0.9497    0.9260    0.9377      1000
       realty     0.9483    0.9530    0.9506      1000
       stocks     0.9023    0.9050    0.9036      1000
    education     0.9698    0.9640    0.9669      1000
      science     0.9030    0.9220    0.9124      1000
      society     0.9404    0.9470    0.9437      1000
     politics     0.9324    0.9240    0.9282      1000
       sports     0.9830    0.9840    0.9835      1000
         game     0.9762    0.9450    0.9604      1000
entertainment     0.9391    0.9720    0.9553      1000
     accuracy                         0.9442     10000
    macro avg     0.9444    0.9442    0.9442     10000
 weighted avg     0.9444    0.9442    0.9442     10000
Confusion Matrix...
[[926   9  38   1   7   5  11   1   1   1]
 [ 10 953   8   1   4  10   5   3   0   6]
 [ 32  21 905   0  25   1  14   0   0   2]
 [  1   0   3 964   3  13   6   0   0  10]
 [  0   4  14   5 922   8  11   1  19  16]
 [  1   5   3  10   7 947  15   0   2  10]
 [  4   5  27   9  14  12 924   1   0   4]
 [  1   3   3   0   1   2   1 984   0   5]
 [  0   3   1   1  31   6   3   1 945   9]
 [  0   2   1   3   7   3   1  10   1 972]]
Time usage: 0:00:08
"""
