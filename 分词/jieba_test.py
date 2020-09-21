# -*- coding: utf-8 -*-
"""
Created on Fri Sep 18 15:49:38 2020

@author: Sally
jieba使用示例
"""
import jieba

text = '南京市长江大桥'
# 精确模式，返回一个可迭代的数据类型
words = jieba.cut(text)
# 所以这个打印的是对象
print(words)
words_list = list(words)

print("精确模式" + str(words_list))

# 表示使用隐马尔可夫模型发现新词
words_HMM = jieba.cut(text, HMM = True)
print("cut with hmm" + str(list(words_HMM)))

# 全模式，输出文本s中所有可能的单词，返回的也是一个可迭代的对象，
# 要想输出需要转换一下，比如转换为List
words2 = jieba.cut(text, cut_all=True)
print("全模式" + str(list(words2)))


# 搜索引擎模式，返回搜索引擎建立索引的分词结果
# 在精确模式的基础上，对长词再次切分，提高召回率，适合用于搜索引擎分词
words3 = jieba.cut_for_search(text)
print("搜索引擎模式" + str(list(words3)))

# 精确模式，返回一个列表类型，建议使用
words_lcut = jieba.lcut(text)
print("精确模式的lcut" + str(list(words_lcut)))

# 全模式，返回一个列表，建议使用
words_lcut_2 = jieba.lcut(text, cut_all = True)
print("全模式的lcut" + str(list(words_lcut_2)))

# 搜索引擎模式，也可以返回列表lcut_for_earch()


# 向分词中增加新词
text2 = '这里有一个个冷，个冷是什么'
word_test = jieba.cut(text2)
print('没添加分词之前：' + str(list(word_test)))

# 添加分词
jieba.add_word('个冷')
word_test = jieba.cut(text2)
print('没添加分词之后：' + str(list(word_test)))

