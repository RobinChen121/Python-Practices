"""
@Python version : 3.12
@Author  : Zhen Chen
@Email   : chen.zhen5526@gmail.com
@Time    : 08/11/2025, 16:59
@Desc    :

"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import gensim.downloader as api
from gensim.models import Word2Vec

# text8 数据集 是从英文维基百科（Wikipedia）中截取的一部分纯文本数据，经过清洗后只包含小写英文字母和空格，
# 常用于测试文本建模或词向量训练.
# corpus 通常是一个句子列表（list of lists），每个句子是一个分词后的单词序列
corpus = api.load("text8")
# Word2Vec 会：遍历语料中的单词序列,习单词之间的上下文关系,
# 将每个单词表示为一个稠密向量（dense vector），即词向量。
model = Word2Vec(corpus)

# fmt: off
words = ['cat', 'dog', 'elephant', 'lion', 'bird', 'rat', 'wolf', 'cow',
         'goat', 'snake', 'rabbit', 'human', 'parrot', 'fox', 'peacock',
         'lotus', 'roses', 'marigold', 'jasmine', 'computer', 'robot',
         'software', 'vocabulary', 'machine', 'eye', 'vision',
         'grammar', 'words', 'sentences', 'language', 'verbs', 'noun',
         'transformer', 'embedding', 'neural', 'network', 'optimization']
# fmt: on

# model.wv.key_to_index 是一个 Python 字典（dict），用于 存储词汇表中每个单词与其对应索引的映射关系
words = [word for word in words if word in model.wv.key_to_index]
word_embeddings = [model.wv[word] for word in words]  # 查看单词对应的词向量
embeddings = np.array(word_embeddings)

tsne = TSNE(n_components=2, perplexity=2)  # 用 tsne 方法降维
embeddings_2d = tsne.fit_transform(embeddings)

# 画图
import matplotlib

matplotlib.use("TkAgg")  # 或者 "Qt5Agg"，具体取决于环境中装了哪个
plt.figure(figsize=(10, 7), dpi=100)
plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], marker="o")
for i, word in enumerate(words):
    plt.text(
        embeddings_2d[i, 0],
        embeddings_2d[i, 1],
        word,
        fontsize=10,
        ha="left",
        va="bottom",
    )
plt.xlabel("t-SNE Dimension 1")
plt.ylabel("t-SNE Dimension 2")
plt.title("Word Embedding Graph (t-SNE with Word2Vec)")
plt.grid(True)
# plt.savefig('embedding.png')
plt.show()
