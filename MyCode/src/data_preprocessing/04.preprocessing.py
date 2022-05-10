import os

import pandas as pd


def merge(path1, path2, path3, mystring):
    print(mystring)
    output = pd.DataFrame(data=None, columns=['comment', 'type_index', 'star_index'])
    data1 = pd.read_csv(path1, sep='\t')
    df1 = data1[['comment', 'type_index', 'star_index']]
    output = output.append(df1)

    data2 = pd.read_csv(path2, sep='\t')
    df2 = data2[['comment', 'type_index', 'star_index']]
    output = output.append(df2)

    data3 = pd.read_csv(path3, sep='\t')
    df3 = data3[['comment', 'type_index', 'star_index']]
    output = output.append(df3)

    output = output.sample(frac=1)
    output.to_csv("{}.csv".format(os.path.join(output_dir, mystring)), index=False, header=False, encoding='utf-8',
                  sep='\t')
    print()


mystring = ["train", "val", "test"]
output_dir = r"../../dataset/预处理后/第四次处理"

for mystr in mystring:
    path1 = r"E:\project\python\文本挖掘\MyCode\dataset\预处理后\第三次处理\中评\{}.csv".format(mystr)
    path2 = r"E:\project\python\文本挖掘\MyCode\dataset\预处理后\第三次处理\好评\{}.csv".format(mystr)
    path3 = r"E:\project\python\文本挖掘\MyCode\dataset\预处理后\第三次处理\差评\{}.csv".format(mystr)
    merge(path1, path2, path3, mystr)
