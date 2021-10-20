# -*- coding: utf-8 -*-
"""
Created on Wed Sep 29 11:51:45 2021

@author: zhen chen

MIT Licence.

Python version: 3.7


Description: 
    
"""
import jieba

# 读取红楼梦的文本内容
txt = open('D:\\红楼梦.txt', 'r', encoding='utf-8').read()
# 运用jieba库对文本内容进行分词
words = jieba.lcut(txt)
# 初始化count字典 用于存放人名出现频率
counts = {}
# 读取红楼梦人名信息
names = open('D:\\人名.txt', 'r', encoding='utf-8').read().split('、')
# 对分词数据进行筛选 将不需要的数据跳过  只保存有效数据
for word in words:
    if len(word) == 1:
        continue
    elif word in '贾母-老太太-老祖宗-史太君'.split('-'):
        word = '贾母'
    elif word in '袭人-蕊珠'.split('-'):
        word = '袭人'
    elif word in '凤姐-琏二奶奶-凤辣子-凤哥儿-凤丫头-琏二嫂子'.split('-'):
        word = '王熙凤'
    elif word in '贾琏-琏二哥哥-琏二爷'.split('-'):
        word = '贾琏'
    elif word in '秦可卿-秦氏-蓉大奶奶-可儿-可卿-兼美'.split('-'):
        word = '秦可卿'            
    elif word in '紫鹃-鹦哥'.split('-'):
        word = '紫鹃'
    elif word in '翠缕-缕儿'.split('-'):
        word = '翠缕'
    elif word in '香菱-甄英莲'.split('-'):
        word = '香菱'
    elif word in '平儿-平姑娘-平妹妹'.split('-'):
        word = '平儿'
    elif word in '豆官-豆童'.split('-'):
        word = '豆官'
    elif word in '薛宝钗-宝钗-蘅芜君-宝姐姐-宝丫头-宝姑娘'.split('-'):
        word = '薛宝钗'
    elif word in '薛宝琴-宝琴'.split('-'):
        word = '薛宝琴'
    elif word in '贾宝玉-宝玉-宝二爷'.split('-'):
        word = '贾宝玉'
    elif word in '林黛玉-林姑娘-黛玉-林妹妹'.split('-'):
        word = '林黛玉'
    elif word in '史湘云-史大姑娘-云姑娘-云妹妹'.split('-'):
        word = '史湘云'      
    if word not in names:
        continue
    counts[word] = counts.get(word, 0) + 1

# 将人名按照次数排序 降序
items = list(counts.items())
# 排序规则 以次数为参考进行排序
items.sort(key=lambda x: x[1], reverse=True)
# print(items)
print('出现次数最多的是:', items[0][0], '出现了：', items[0][1], '次')
print('出现次数最少的是:', items[-1][0], '出现了：', items[-1][1], '次')
for item in items:
    print(item[0], '出现了：', item[1], '次')