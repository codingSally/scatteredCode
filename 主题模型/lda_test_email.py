# -*- coding: utf-8 -*-
"""
Created on Sun Sep 20 19:13:42 2020

@author: Sally
希拉里垃圾邮件分类
"""

import numpy as np
import pandas as pd
import re
from gensim import corpora, models, similarities
import gensim
from nltk.corpus import stopwords
import jieba


# 清理邮件内容
def clean_email_text(text):
    # 换行
    text = text.replace('\n', " ")
    # 把"-"的两个单词， 分开。（比如：july-edu ==> july edu）
    text = re.sub(r"-", "", text)
    # 日期， 对主题模型没有什么意义
    text = re.sub(r"\d+/\d+/\d+", "", text)
    # 时间，没意义
    text = re.sub(r"[0-2]?[0-9]:[0-6][0-9]", "", text)
    # 邮件地址，没意义
    text = re.sub(r"[\w]+@[\.\w]+", "", text) 
    # 网址，没意义
    text = re.sub(r"/[a-zA-Z]*[:\//\]*[A-Za-z0-9\-_]+\.+[A-Za-z0-9\.\/%&=\?\-_]+/i", "", text)
    
    #  以防还有其他特殊字符（数字）等等，我们直接把他们loop一遍，过滤掉
# =============================================================================
#     优化代码，pyhton中字符串也是不可变对象，执行 + 操作会创建大量对象
#     pure_text = ''
#     # 以防还有其他特殊字符（数字）等等，我们直接把他们loop一遍，过滤掉
#     for letter in text:
#         # 只留下字母和空格
#         if letter.isalpha() or letter==' ':
#             pure_text += letter
#     # 再把那些去除特殊字符后落单的单词，直接排除。
#     # 我们就只剩下有意义的单词了。
#     text = ' '.join(word for word in pure_text.split() if len(word)>1)
# =============================================================================
    pure_text = []
    for letter in text:
        # 只留下字母和空格
        # isalpha求字符串中是否只包含字母
        if letter.isalpha() or letter == '':
            pure_text.append(letter)
    # 再把那些去除特殊字符后落单的单词，直接排除。
    # 我们就只剩下有意义的单词了。
    # print('拼接后的list：\n', pure_text)
    # text = ''.join(word for word in pure_text if len(word) > 1) 
    text = ''.join(word for word in pure_text) 
    
    return text

# 数据加载
df = pd.read_csv('.\Emails.csv')
# 原始邮件中又许多Nan的值，直接drop
df = df[['Id', 'ExtractedBodyText']].dropna()

docs = df['ExtractedBodyText']
# print(docs)
docs = docs.apply(lambda x : clean_email_text(x))
# print('转换后： \n' ,docs)

# docs[Series] 转换为list
doclist = docs.values
print('邮件数量： \n', len(doclist))

# 将邮件内容拼接成文本
texts = [[word for word in jieba.cut(doc)] for doc in doclist]
print('拼接后的文本： \n', texts)

# 构建语料库，将文本ID化
dictionary = corpora.Dictionary(texts)
print('字典表： \n', dictionary)

# 把文档Doc变成一个稀疏向量
corpus = [dictionary.doc2bow(text) for text in texts]

# 将每一篇邮件ID化
print("第一封邮件ID化后的结果为： \n", corpus[0], '\n')

"""训练LDA模型"""
# 所有主题单词的分布
lda = gensim.models.ldamodel.LdaModel(corpus = corpus, id2word = dictionary, num_topics = 10)

# 每一行包含了主题词和主题词权重
print("主题词和对应的权重： \n", lda.print_topics(num_topics=10, num_words=10))

# topic中的重要单词
for topic in lda.print_topics(num_topics=5):
    print(topic)
    

print('输出一些主题信息')
print(lda.get_document_topics(corpus[0]))
print(lda.get_document_topics(corpus[1]))
print(lda.get_document_topics(corpus[2]))
