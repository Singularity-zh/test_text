#!/usr/bin/env Python
# coding=utf-8
import os
import re
import time

import numpy as np
import pandas as pd

# dic_cha = {"1": "水润保湿面膜", "2": "胶原蛋白面膜", "3": "美白保湿黑炭面膜", "4": "毛孔紧致黑面膜", "5": "水润安瓶面膜", "6": "维C安瓶面膜", "7": "维生素面膜",
#            "8": "茶树精油面膜", "9": "珍珠蛋白面膜"}


""""

type_index_list:代表不同的小类，对应每个文件的第一序号
star_index_list：代表评价的等级
    0-好评
    1-中评
    2-差评
"""
#

my_str_dic = {"好评": "0", "中评": "1", "差评": "2"}


def merge(file_path, output_dir, my_str):
    file_name = os.path.splitext(os.path.basename(file_path))[0]
    type_index = file_name.split(".")[0]
    output_name = os.path.join(output_dir, file_name)
    print(output_name)
    print(file_name)
    data = pd.read_csv(file_path, sep='\t')
    count = data.shape[0]
    type_index_list = [type_index for i in range(count)]
    star_index = my_str_dic.get(my_str)
    star_index_list = [star_index for i in range(count)]
    data.insert(loc=len(data.columns), column="type_index", value=type_index_list)
    data.insert(loc=len(data.columns), column="star_index", value=star_index_list)
    data.to_csv("{}.csv".format(output_name), index=False, encoding="utf-8", sep="\t")


# file = r"../../dataset/预处理后/第一次处理/好评/1.美迪惠尔面膜-京东-水润面膜-好评.csv"
# merge(file)


my_str_list = ["好评", "中评", "差评"]
for my_str in my_str_list:
    output_dir = r"../../dataset/预处理后/第二次处理/{}/".format(my_str)
    dir_path = r"../../dataset/预处理后/第一次处理/{}/".format(my_str)
    for root, dirs, files in os.walk(dir_path):
        for file in files:
            file_path = os.path.join(root, file)
            print(file_path)
            merge(file_path, output_dir, my_str)

"""
差评：
{"1":"水润保湿面膜","2":"胶原蛋白面膜","3":"美白保湿黑炭面膜","4":"毛孔紧致黑面膜","5":"水润安瓶面膜","6":"维C安瓶面膜","7":"维生素面膜","8":"茶树精油面膜","9":"珍珠蛋白面膜"}

"""
