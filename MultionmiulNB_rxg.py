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
from functools import reduce
import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer,TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from scipy import sparse

class MultionmiulNB_rxg():
    def __init__(self,alpha=1,fit_prior=False,class_prior=None):
        self.alpha = alpha
        self.fit_prior = fit_prior
        self.class_prior = class_prior
    def fit(self,texts,class_label):
        words_list = set()  # 存放没有重复的全部词，也是词向量的长度
        for text in texts:
            words_list = set(text) | words_list
        self.words_list = list(words_list)
        len_words_list = len(self.words_list)
        texts_list = []  # 每个文本的词向量列表
        for text in texts:
            word_dic = dict()
            for word in text:
                if word not in word_dic:
                    word_dic[word] = 1
                else:
                    word_dic[word] += word_dic[word]
            texts_vec = np.zeros(len_words_list)  # 构建一个空的列表用来存放词向量，长度为全词没有重复的
            for i in range(len_words_list):
                if self.words_list[i] in text:
                    texts_vec[i] += word_dic[self.words_list[i]]
            texts_list.append(texts_vec)

            p0_vec = np.ones(len_words_list)
            p0_vec_sum = 0
            p1_vec = np.ones(len_words_list)
            p1_vec_sum = 0
            for i in range(len(texts)):
                if class_label[i] == 0:
                    p0_vec += texts_vec
                    p0_vec_sum = sum(p0_vec)
                else:
                    p1_vec += texts_vec
                    p1_vec_sum = sum(p1_vec)
            if self.fit_prior:
                self.p0_v = self.class_prior[0] * p0_vec / p0_vec_sum
                self.p1_v = self.class_prior[1] * p1_vec / p1_vec_sum
            else:
                c0 = class_label.count(list(set(class_label))[0])
                c1 = class_label.count(list(set(class_label))[1])
                self.p0_v = c0 * p0_vec / p0_vec_sum
                self.p1_v = c1 * p1_vec / p1_vec_sum
        return texts_list
    def predict(self,test):
        test_word_dic = dict()
        for word in test:
            if word not in test_word_dic:
                test_word_dic[word] = 1
            else:
                test_word_dic[word] += test_word_dic[word]
        if self.alpha == 1:
            test_p = np.ones(len(self.words_list))
        elif self.alpha == 0:
            test_p = np.zeros(len(self.words_list))
        else:
            raise 'the default of alpha is 0 or 1'
        for i in range(len(self.words_list)):
            if self.words_list[i] in test_word_dic:
                test_p[i] += test_word_dic[self.words_list[i]]
        p0 = self.p0_v * test_p
        p0 = reduce(lambda x,y:x*y,p0)
        p1 = self.p1_v * test_p
        p1 = reduce(lambda x,y:x*y,p1)

        if p0==0 and p1 == 0:
            raise '此数据集拉普拉斯平滑参数为0会报错'
        p00 = p0/(p0+p1)
        p11 = p1/(p0+p1)

        return p00,p11
    def prob_(self):
        '''
        :return:  返回二分类词向量概率列表
        '''
        return self.p0_v,self.p1_v


#此处的类并不是直接计算分类结果，而是将训练集和预测集分别处理成维度相同的数据个数，可以用各种分类器进行分类
class rxg_tfidf():
    def __init__(self):
        pass
    def fit(self,texts):
        '''
        :param texts: 训练数据集
        :return:
        '''
        count_vec = CountVectorizer()
        word_count = count_vec.fit_transform(texts)    #统计词频   此时的格式为sparse
        tf_idf = TfidfTransformer()
        word_tfidf =tf_idf.fit_transform(word_count)   #计算tfidf  此时的格式为sparse
        train_data_matrix = sparse.csr_matrix(word_tfidf).A

        self.word_vec = count_vec.get_feature_names()   #词向量的每个词的顺序
        self.idf_ = tf_idf.idf_
        return train_data_matrix
    def predict(self,test):
        count_vec = CountVectorizer()
        test_word_count = count_vec.fit_transform(test)
        test_word_count = sparse.csr_matrix(test_word_count).A[0]     #得到当前的test的词频向量
        test_word = count_vec.get_feature_names()                  #得到当前test的每个词的顺序
        # test_c_w = list(zip(test_word_count,test_word))

        test_vec = np.zeros(len(self.word_vec))
        for i in range(len(self.word_vec)):
            if self.word_vec[i] in test_word:
                test_vec[i] += test_word_count[test_word.index(self.word_vec[i])]
                test_vec[i] = test_vec[i] / self.idf_[i]
        return test_vec


if __name__ == '__main__':
    '''
    texts = [['the', 'authorized', 'version', 'of', 'the', 'Bible'], ['you', 'are', 'bosset'], ['what', 'dog']]
    test = ['the','authorized']
    class_label = [1,0,1]
    rf = MultionmiulNB_rxg(alpha=1,fit_prior=True,class_prior=[0.8,0.2])
    texts_list = rf.fit(texts=texts,class_label=class_label)
    p0,p1 = rf.predict(test)
    print(p0,p1)
    '''
    corpus = ["I come to China to travel","This is a car polupar in China","I love tea and Apple ","The work is to write some papers in science"]
    class_label = [1, 0, 1,1]
    test = ['the authorized China polupar']
    rf = rxg_tfidf()
    rf.fit(corpus)
    rf.predict(test)
