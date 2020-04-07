#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__title__ = ''
__author__ = '任晓光'
__mtime__ = '2020/3/24'
# code is far away from bugs with the god animal protecting
    I love animals. They taste delicious.
              ┏┓      ┏┓
            ┏┛┻━━━┛┻┓
            ┃      ☃      ┃
            ┃  ┳┛  ┗┳  ┃
            ┃      ┻      ┃
            ┗━┓      ┏━┛
                ┃      ┗━━━┓
                ┃  神兽保佑    ┣┓
                ┃　永无BUG！   ┏┛
                ┗┓┓┏━┳┓┏┛
                  ┃┫┫  ┃┫┫
                  ┗┻┛  ┗┻┛
"""
import os
import pandas as pd
import re
from sklearn.feature_extraction.text import CountVectorizer
from scipy import sparse
def read_data(path):
    part1_list = os.listdir(path)   #part根目录
    data = []    #把每一个txt文件 放到一个list中
    label = []
    for path1 in part1_list:
        path2 = os.path.join(path,path1)
        part3_list = os.listdir(path2)
        for part3 in part3_list:
            data.append(os.path.join(path2,part3))
            isfalse = re.match('spmsgc',part3)
            if isfalse is None:
                label.append(1)
            else:
                label.append(0)
    return data,label


if __name__ == '__main__':
    # path = r'D:\算法实现\navie_bayes\data\lingspam_public\bare'
    # read_data(path)

    pa = r'D:\算法实现\navie_bayes\data\lingspam_public\bare\part1\3-1msg1.txt'
    f = open(pa)
    data = f.readlines()

    co = CountVectorizer(stop_words='english')
    temp = co.fit_transform(data)
    print(len(co.get_feature_names()))

    # print(sparse.csr_matrix(temp))