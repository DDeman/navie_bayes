#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__title__ = ''
__author__ = '任晓光'
__mtime__ = '2020/3/23'
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
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score,confusion_matrix
class GaussianNB_rxg():
    def __init__(self):
        pass
    def fit(self,x_train):
        '''
        :param x_train: 最后一列为带标签列的数据
        :return:
        '''
        x_train = pd.DataFrame(x_train)
        y_train = pd.DataFrame(x_train.iloc[:,-1],columns=['class'])   #取x_train的最后一列是class
        self.class_label = y_train['class'].value_counts().index
        feature_names = list(x_train)   #获得x_train的列标签
        mean = []
        std = []
        for i in self.class_label:
            item = x_train[x_train['class'] == i]
            m = np.mean(item)
            s = np.var(item)
            mean.append(m)
            std.append(s)
        self.mean_ = pd.DataFrame(mean, columns=feature_names[:-1], index=self.class_label)
        self.std_ = pd.DataFrame(std, columns=feature_names[:-1], index=self.class_label)
    def predict(self,test):
        '''
        :param test: 为没有标签列的数据,输入数据类型为list或者dataframe
        :return: 预测的test的类
        '''
        test = pd.DataFrame(test)
        self.y_pred = []
        for j in range(test.shape[0]):
            iset = test.iloc[j, :].tolist()
            print(iset)
            iprob = np.exp(-(iset - self.mean_) ** 2 / (self.std_ * 2)) / (np.sqrt(2 * np.pi * self.std_))
            temp = []
            for i in range(len(self.class_label)):
                prob = 1
                for pp in iprob.iloc[i]:
                    prob *= pp
                temp.append(prob)
                #     print(temp)
            cls = self.class_label[np.argmax(temp)]
            self.y_pred.append(cls)
        return self.y_pred
    def acc_score(self,y_true):
        return accuracy_score(y_true,self.y_pred)
    def confusion_(self,y_true):
        return confusion_matrix(y_true,self.y_pred)


if __name__ == '__main__':
    # gass_nb = GaussianNB_rxg()
    data = load_iris()
    x = data.data
    y = data.target
    feature_names = data.feature_names
    labels_names = data.target_names
    x_df = pd.DataFrame(x, columns=feature_names)
    y_df = pd.DataFrame(y, columns=['class'])
    data = pd.concat([x_df, y_df], axis=1)
    x_train, x_test, y_train, y_test = train_test_split(data.iloc[:, 0:4], data.iloc[:, 4:5], test_size=0.3)
    x_train = pd.concat([x_train, y_train], axis=1)
    x_test = pd.concat([x_test, y_test], axis=1)
    train = x_train.reset_index(drop=True)
    test = x_test.reset_index(drop=True)
    test = test.iloc[:,0:-1]
    # print(train)
    gass_nb = GaussianNB_rxg()
    gass_nb.fit(train)
    pred = gass_nb.predict(test)
    # print(y_test['class'].tolist())
    print(pred)
    print(y_test['class'].tolist())
    acc = gass_nb.acc_score(y_test['class'].tolist())
    print(acc)
