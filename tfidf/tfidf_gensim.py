# -*- coding: utf-8 -*-
"""
Created on Fri Sep 18 16:47:26 2020

@author: Sally
Gensim, TF-IDF使用
"""
import jieba
from gensim import corpora, models, similarities

# 加载数据
wordslist = ["我非常喜欢看电视剧","我非常喜欢旅行","我非常喜欢吃苹果", '我非常喜欢跑步'] 

# 分词
textTest = [[word for word in jieba.cut(words)] for words in wordslist]
print('分词结果：\n', textTest)

# 生成字典
dictionary = corpora.Dictionary(textTest)
print("字典：\n", dictionary)
# 字典词频，{单词ID, 在多少文档中出现}
print('单词词频：', dictionary.dfs)
# 文档数量
print('文档数量：\n', dictionary.num_docs)
# 所有词的个数
print('所有词的个数：\n', dictionary.num_pos)
feaures = dictionary.token2id.keys()
print('生成的特征是：\n', feaures)
print('生成特征和ID的映射：\n', dictionary.token2id)
featurenum = len(dictionary.token2id.keys())
print('生成特征数量：\n', featurenum)


# 生成语料:句子的分词结果中，每个'特征'出现的次数
# 标记文本为向量
corpus = [dictionary.doc2bow(text) for text in textTest]
print('生成的语料：\n', corpus)

""" 计算语料的TFIDF """ 
# 训练TFIDF模型
tfidf_model = models.TfidfModel(corpus, dictionary = dictionary)
# 只要记录BOW矩阵的非零元素个数(num_nnz)
print('TFIDF模型: \n', tfidf_model)

# 得到语料的TFIDF值
corpus_tfidf = tfidf_model[corpus]
print('corpus = ', corpus)
print('corpus_tfidf=',corpus_tfidf )
print('='*10 + '转换整个语料库' + '='*10)
for doc in corpus_tfidf:
    print(doc)
    
# 生成余弦相似度索引, 使用SparseMatrixSimilarity()，可以占用更少的内存和磁盘空间。
index = similarities.SparseMatrixSimilarity(corpus_tfidf, num_features=featurenum) 

""" 对于新的句子，生成 BOW向量，与之前的句子计算相似度 """
test = jieba.lcut('我喜欢看电视剧')
print('测试分词结果：\n', test)
# 生成BOW向量
vec = dictionary.doc2bow(test)
print('BOW向量：\n', vec)
# 计算TFIDF向量
test_vec = tfidf_model[vec]
print('test vec = ', test_vec)
# 返回test_vec 和 训练语料中所有文本的余弦相似度，返回结果是个numpy数组
print('相似文本：\n' , index.get_similarities(test_vec))
