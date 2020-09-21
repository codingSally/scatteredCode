# -*- coding: utf-8 -*-
"""
Created on Fri Sep 18 22:05:35 2020

@author: Sally
词云测试
"""

from wordcloud import WordCloud
import jieba
import jieba.analyse
# import jieba.posseg as pseg
import matplotlib.pyplot as plt


sentence = '王者荣耀典韦连招是使用一技能+大招+晕眩+二技能+普攻，这套连招主要用于先手强开团，当发现对面走位失误或撤退不及时，我们就可以利用一技能的加速，配合大招减速留住对手，协同队友完成击杀。当对方站位较集中时，我们同样可以利用“一技能+大招+晕眩”进行团控和吸收伤害。在吸收伤害的同时我们还可以利二技能打出不错的输出。这套连招重要的是把握时机，要有一夫当关，万夫莫开之势。缺点是一技能的强化普攻和解除控制的效果会被浪费。'

# 提取句子关键字
keywords = jieba.analyse.textrank(sentence, topK=20, withWeight=True) 

# 实例化词云
wc = WordCloud(font_path = 'fonts/simsun.ttc')

# keywords, 需要转换为字典传入wc
frequence_dic = {w : weight for w, weight in keywords}
print(frequence_dic)

# 生成词云
wc.generate_from_frequencies(frequence_dic)

plt.imshow(wc)

"""设备指纹的应用想象"""
# 1. 品牌、型号等单品牌词云展示，对比黑白设备的常用手机
# 2. 常用APP的词云展示，对比黑白设备的常用APP
# 3. 其他属性
# 4. 我觉得这个实现的时候，就将各类样本的如品牌抽取出来，做一个展示对比就可以


