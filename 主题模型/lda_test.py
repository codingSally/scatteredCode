# -*- coding: utf-8 -*-
"""
Created on Fri Sep 18 22:54:42 2020

@author: Sally
LDA的一个简单示例
LDA主题模型，应用于文本关键字提取
"""

import jieba
from gensim import corpora, models
# import jieba.posseg as pseg

# 加载数据
wordslist = ["我非常喜欢看电视剧","我非常喜欢旅行","我非常喜欢吃苹果", '我非常喜欢跑步', '王者荣耀春季赛开战啦']

# 分词的时候不被切掉
jieba.add_word('王者荣耀', tag='n')
 
# 分词
textTest = [[word for word in jieba.cut(words)] for words in wordslist]
print('分词：\n', textTest)
# 生成字典
dictionary = corpora.Dictionary(textTest)
print('字典中的字和ID的对应关系：\n', dictionary.token2id)
# =============================================================================
# print('字典：', dictionary)
# print('单词词频:', dictionary.dfs)  # 字典词频，{单词id，在多少文档中出现}
# print('文档数目:', dictionary.num_docs)  # 文档数目
# print('所有词的个数:', dictionary.num_pos)  # 所有词的个数
# featurenum = len(dictionary.token2id.keys())
# print('featurenum', featurenum)
# =============================================================================

# 生成语料 
corpus = [dictionary.doc2bow(text) for text in textTest]
print('语料：\n', corpus)

# 训练LDA模型
lda = models.ldamodel.LdaModel(corpus = corpus, id2word = dictionary, num_topics = 2)

# 完全不知道输出的是一些什么东西？？？
for topicid, topic in lda.print_topics(num_words=5):
    print('主题' + str(topicid) + '\n' , topic)

# 主题推断
print('主题推断：\n', lda.inference(corpus))
text5 = '我喜欢看王者荣耀KPL挑战赛'

# bow向量
# print('分词结果：\n', list(jieba.cut(text5)))
bow = dictionary.doc2bow([word for word in jieba.cut(text5)])
# bow向量和分词的关系是什么，为什么生成的bow向量不是分词在语料中的出现次数
# bow向量个数和分词的个数都不一致？？？需要继续深入研究
print('bow向量： \n', bow)

# 主题推断结果
inference_result = lda.inference([bow])[0]
for e, value in enumerate(inference_result[0]):
    print('主题{} 推断值{} \n'.format(e, value))\
       
# 测试单词        
# 得到向量ID
word = '王者荣耀'
word_id = dictionary.doc2idx([word])[0]
print('单词向量ID: \n', word_id)

# 得到指定单词与主题关系
print('主题与单词的对应关系 \n', lda.get_term_topics(word_id))
for i in lda.get_term_topics(word_id):
    print('{}与主题{}的关系值为{}%'.format(word, i[0], i[1]*100))

# 查看主题0的重要词汇
print(lda.get_topic_terms(0, topn=10))













