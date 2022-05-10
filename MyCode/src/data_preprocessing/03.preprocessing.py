import logging
import math
import os
import numpy as np
import pandas as pd

my_str_list = ["好评", "中评", "差评"]
for my_str in my_str_list:

    dir_path = r"../../dataset/预处理后/第二次处理/{}/".format(my_str)
    output_dir = r"../../dataset/预处理后/第三次处理/{}/".format(my_str)

    factor_train = 0.6
    train_output = pd.DataFrame(data=None,
                                columns=['user_name', 'is_plus', 'star_num', 'comment', 'commodity_type', 'type_index',
                                         'star_index'])
    val_output = pd.DataFrame(data=None,
                              columns=['user_name', 'is_plus', 'star_num', 'comment', 'commodity_type', 'type_index',
                                       'star_index'])
    test_output = pd.DataFrame(data=None,
                               columns=['user_name', 'is_plus', 'star_num', 'comment', 'commodity_type', 'type_index',
                                        'star_index'])

    count = 0

    for root, dirs, files in os.walk(dir_path):
        for file in files:
            print(file)
            file_name = os.path.splitext(file)[0].split('.')[1].split("-")[-2]
            file_path = os.path.join(root, file)
            print(file_path)
            csv_data = pd.read_csv(file_path, sep='\t')
            data_count = csv_data.shape[0]
            print(data_count)
            indices = np.arange(data_count)
            np.random.seed(20220505)
            np.random.shuffle(indices)

            # 划分训练集
            train_size = int(factor_train * data_count)
            train = csv_data.iloc[indices[:train_size]]
            #
            # # 划分验证集
            test_val_size = math.ceil((data_count - train_size) / 2)
            val = csv_data.iloc[indices[train_size:train_size + test_val_size]]
            #
            # # 划分测试集
            test = csv_data.iloc[indices[train_size + test_val_size:data_count]]
            #
            train_output = train_output.append(train)
            val_output = val_output.append(val)
            test_output = test_output.append(test)
            print()

            # logging.info(
            #     "当前类别为：{}，图片总数为：{}，划分训练集的数量：{}，划分验证集的数量：{}，划分测试集的数量：{}".format(file_name, data_count, train_size,
            #                                                                    test_val_size,
            #                                                                    data_count - train_size - test_val_size))
            count += data_count

    train_output = train_output.sample(frac=1)
    train_output.to_csv("{}.csv".format(os.path.join(output_dir, "train")), index=False, encoding='utf-8', sep='\t')

    val_output = val_output.sample(frac=1)
    val_output.to_csv("{}.csv".format(os.path.join(output_dir, "val")), index=False, encoding='utf-8', sep='\t')
    # #
    #
    test_output = test_output.sample(frac=1)
    test_output.to_csv("{}.csv".format(os.path.join(output_dir, "test")), index=False, encoding='utf-8', sep='\t')
    #
    # logging.info("差评数据集划分完毕，数据集容量为：{}".format(count))
