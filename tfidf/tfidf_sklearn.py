# -*- coding: utf-8 -*-
"""
Created on Fri Sep 18 16:46:49 2020

@author: Sally
sklearn中，TF-IDF使用
"""

import jieba
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer

# 加载数据
wordslist = ["我非常喜欢看电视剧，电视剧","我非常喜欢旅行","我非常喜欢吃苹果", '我非常喜欢跑步', '我爱旅行']

# 分词
textTest = [' '.join(jieba.cut(words, HMM = True)) for words in wordslist]
print('分词结果：' + '\n' + str(textTest))
# 等价于下面的
print('分词结果: \n', textTest)

# 统计词频
vectorizer = CountVectorizer()
count = vectorizer.fit_transform(textTest)
# 特征
print("特征名称【文本中所有关键字】：" + '\n' +  str(vectorizer.get_feature_names()))
# 字典表
# 训练出来的字典表，个人理解数字没有权重含义，只是index(特征索引)，用于构建矩阵
print("训练出来的字典表" + '\n' +  str(vectorizer.vocabulary_))
# print(vectorizer.vocabulary)

# 词频统计矩阵【每个句子中，对应关键字出现的次数】
#  print(count)
print("词频矩阵" + '\n' +  str(count.toarray())) 

""" 计算TF-IDF """ 
# TfidfTransformer是统计CountVectorizer中每个词语的tf-idf权值
transformer = TfidfTransformer()

# TFIDF矩阵矩阵
tfidf_matrix = transformer.fit_transform(count)
print("TF-IDF矩阵" + '\n' + str(tfidf_matrix.toarray()))


""" TfidfVectorizer可以把CountVectorizer, TfidfTransformer合并起来，直接生成tfidf值 """ 
tfidf_vec = TfidfVectorizer()
tfidf_matrix_2 = tfidf_vec.fit_transform(textTest)

print("关键字：" + '\n' +  str(tfidf_vec.get_feature_names()))
print("字典表" + '\n' +  str(tfidf_vec.vocabulary_))
print("词频矩阵" + '\n' +  str(tfidf_matrix_2.toarray())) 

""" 单词与单词之间的余弦相似度 """
from sklearn.metrics.pairwise import cosine_similarity
print("余弦相似度=\n", cosine_similarity(tfidf_matrix_2, tfidf_matrix_2))

# 测试数据
test = [' '.join(jieba.cut("我爱跑步"))]
# 得到tfidf vec
# 注意这里直接用训练好的tfidf_vec，做transform就好了，不需要fit_transform了
# fit表示训练；transform表示转换； fit_transform表示训练并转换
test_vec = tfidf_vec.transform(test)
print("余弦相似度=\n", cosine_similarity(test_vec, tfidf_matrix_2))


"""有感而发：设备指纹中的应用"""
# 采用分词，对于一批量的APP,形成词频矩阵，计算设备APP见的相似性
# 相似性大|相似性大且在一个网段
# 离线可以计算，但是如果线上可以直接使用训练好的词频矩阵吗？？？
# 看case是可以的，可以计算新来的数据，和哪个相似
# 那么在设备指纹里面，如果抓住了一批黑的设备APP，那么可以对新来的做相似性判断



