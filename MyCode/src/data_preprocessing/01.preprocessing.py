#!/usr/bin/env Python
# coding=utf-8
import os
import re
import time

import numpy as np
import pandas as pd


def stopwordslist():
    stopwords = [line.strip() for line in
                 open('../../dataset/stopwords/cn_stopwords.txt', encoding='UTF-8').readlines()]
    return stopwords


stop_words = stopwordslist()
# str_mystop =r'\b(' + r'|'.join(stop_words) + r')\b'
# print(str_mystop)



# file_path = r"../../dataset/面膜数据/中评/1.美迪惠尔面膜-京东-水润面膜-中评.csv"

def getdata(file_path, output_dir):
    file_name = os.path.splitext(os.path.basename(file_path))[0]
    output_name = os.path.join(output_dir, file_name)
    print(file_name)
    data = pd.read_csv(file_path)
    shape = data.shape
    print(shape[0])
    data.fillna("", inplace=True)
    user_name = data["用户名"]
    is_plus = data["会员"].map(lambda x: 1 if x == 'PLUS会员' else 0)
    star_num = data["星级"].map(lambda x: regular_match(x))
    comment = data["评论"].map(lambda x: data_clean(x))
    commodity_type = data["类型1"].str.cat(data["类型2"])
    data_frame = pd.DataFrame({"user_name": user_name, "is_plus": is_plus, "star_num": star_num, "comment": comment,
                               "commodity_type": commodity_type})
    print(sum(data_frame["commodity_type"].value_counts()))
    data_frame.to_csv("{}.csv".format(output_name), index=False, encoding="utf-8", sep="\t")


def data_clean(x):
    pattern = re.compile(r'\b(' + r'|'.join(stop_words) + r')\b')
    x = pattern.sub('', x)
    x = re.sub(r'[’!\"#$%&\'()*+-/<=>?@?:：★、…【】《》？“”‘’！\[\\\]^_`{|}~]+', '', x)
    x = re.sub(r"[。]{2,}", "。", x)
    x = re.sub(r"[，]{2,}", "，", x)
    x = re.sub(r"[,]{2,}", ",", x)
    x = re.sub(r'\s+', "", x)
    x = re.sub(r'\d+', "", x)
    # x = re.sub(r'[a-zA-Z]', "", x)
    return x


def get_timestamp(timeStr):
    if timeStr == "":
        return 0
    timeArray = time.strptime(timeStr, '%Y/%m/%d %H:%M')  # 按照对应的格式转换为时间数组time.strptime()
    timeStamp = int(time.mktime(timeArray))  # 转换成整形时间戳 time.mktime()
    return timeStamp


def regular_match(str):
    re_str = r"(?<=star star)\d"
    star = re.findall(re_str, str)
    return star[0]


# getdata(file_path)

my_str_list = ["好评", "中评", "差评"]
for my_str in my_str_list:

    dir_path = r"../../dataset/面膜数据/{}/".format(my_str)

    output_dir = r"../../dataset/预处理后/第一次处理/{}/".format(my_str)
    for root, dirs, files in os.walk(dir_path):
        for file in files:
            file_path = os.path.join(root, file)
            # print(file_path)
            getdata(file_path, output_dir)

# 用户名,会员,星级,评论,采集时间,类型1,评论时间1,类型2,评论时间2,评论时间3,评论时间4

# 用户名,会员,星级,评论,采集时间,类型1,类型2,评论时间1,评论时间2,评论时间3
# 用户名,会员,星级,评论,采集时间,类型1,类型2,评论时间1,评论时间2,评论时间3

""""
1.美迪惠尔面膜-京东-水润面膜-中评
930
930
10.美迪惠尔面膜-京东-燕窝蛋白弹力乳液面膜-中评
18
18
11.美迪惠尔面膜-京东-透明质酸原液水感面膜-中评
2
2
2.美迪惠尔面膜-京东-胶原蛋白-中评
1000
1000
3.美迪惠尔面膜-京东-美白保湿黑面膜-中评
1000
1000
4.美迪惠尔面膜-京东-毛孔紧致黑面膜-中评
980
980
5.美迪惠尔面膜-京东-水润舒缓安瓶面膜-中评
44
44
6.美迪惠尔面膜-京东-维他命C净透安瓶面膜-中评
33
33
7.美迪惠尔面膜-京东-维生素VC面膜面膜-中评
592
592
8.美迪惠尔面膜-京东-茶树精油面膜-中评
617
617
9.美迪惠尔面膜-京东-珍珠蛋白净透乳液面膜-中评
21
21
"""

"""
1.美迪惠尔面膜-京东-水润面膜-差评
997
997
2.美迪惠尔面膜-京东-胶原蛋白-差评
679
679
3.美迪惠尔面膜-京东-美白保湿竹炭黑面膜-差评
960
960
4.美迪惠尔面膜-京东-毛孔紧致黑面膜-差评
950
950
5.美迪惠尔面膜-京东-水润舒缓安瓶面膜-差评
64
64
6.美迪惠尔面膜-京东-维他命C净透安瓶面膜-差评
53
53
7.美迪惠尔面膜-京东-维生素VC面膜-差评
655
655
8.美迪惠尔面膜-京东-茶树精油面膜-差评
514
514
9.美迪惠尔面膜-京东-珍珠蛋白净透乳液面膜-差评
15
15

"""
