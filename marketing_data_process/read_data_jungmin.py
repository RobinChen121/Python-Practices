"""
Python version: 3.12.7
Author: Zhen Chen, chen.zhen5526@gmail.com
Date: 2025/12/15 17:59
Description: 
    

"""
import sys
import pandas as pd
import re
import time
import os

# def count_dictionary_words(text, dictionary):
#     text = text.lower()
#     counts = {}
#
#     for word in dictionary:
#         w = word.lower()
#         # 使用正则，确保是完整词或完整短语
#         pattern = r'\b' + re.escape(w) + r'\b' # escape() 把字符串里的正则特殊字符全部“转义”成普通字符, \b 表示边界
#         matches = re.findall(pattern, text) # 在 text 中找到所有符合 pattern 的匹配项，返回 list
#         counts[word] = len(matches)
#
#     return counts


# 94 words
dictionary = [
    'magnificent', 'excellent', 'awesome', 'fantastic', 'perfect', 'amazing', 'outstanding',
    'wonderful', 'fabulous', 'lovable', 'great', 'very good', 'wise', 'terrific', 'joyful',
    'exciting', 'smart', 'positive', 'delightful', 'valuable', 'attractive', 'healthy', 'cheerful',
    'beneficial', 'enjoyable', 'desirable', 'pro', 'helpful', 'favourable', 'superior', 'pleasant',
    'relaxing', 'worthwhile', 'likable', 'appealing', 'useful', 'good', 'wholesome', 'calming',
    'commendable', 'nice', 'safe', 'agreeable', 'reasonable', 'acceptable', 'satisfactory', 'okay',
    'adequate', 'neutral', 'average', 'tolerable', 'mediocre', 'questionable', 'imperfect', 'objectionable*',
    'boring', 'foolish', 'ridiculous', 'sorrowful', 'inappropriate', 'troublesome', 'dislikable', 'unhealthy',
    'upsetting', 'saddening', 'inferior', 'con', 'unsafe', 'annoying', 'offensive', 'angering',
    'irritating', 'stupid', 'bad', 'frightening', 'harmful', 'dangerous', 'negative', 'undesirable',
    'disturbing', 'appalling', 'depressing', 'useless', 'disgusting', 'terrifying', 'sickening', 'horrible',
    'awful', 'gruesome', 'dreadful', 'terrible', 'repulsive', 'worthless', 'hateful']

file_address = ''
if sys.platform != 'Windows':
    file_address += '/Users/zhenchen/Documents/working paper datasets/Jungmin/5 score/'
file_name = 'All_Beauty_5.json.gz'


def count_words(text):
    text = str(text).lower()
    counts = {'Length': len(text.split())}
    counts.update({w: 0 for w in dictionary})

    # 预编译正则，节约时间
    # escape() 把字符串里的正则特殊字符全部“转义”成普通字符, \b 表示边界
    pattern = r'\b(' + '|'.join(re.escape(w.lower()) for w in dictionary) + r')\b'
    matches = re.findall(pattern, text)

    for m in matches:
        counts[m] += 1

    return counts


start = time.time()
file_path = file_address + file_name
size_MB = os.path.getsize(file_path) / 1024 ** 2
if size_MB < 500:
    df = pd.read_json(file_path, compression='gzip', lines=True)
    word_counts = df['reviewText'].apply(count_words)
    word_counts = pd.DataFrame(word_counts.to_list())
    # word_total = word_counts.sum()

    file_name_output = file_name[:-8] + '_output.json.gz'
    # df_counts = pd.DataFrame.from_dict(word_counts, orient='index', columns=[file_name[:-8]])
    output_address = file_address + 'output/'
    word_counts.to_json(output_address + file_name_output, orient='records', lines=True, compression='gzip')
    end = time.time()
    print('running time is {} seconds'.format(end - start))
else:
    chunk_size = 1_000_000
    index = 1

    reader = pd.read_json(
        file_path,
        compression="gzip",
        lines=True,
        chunksize=chunk_size
    )

    for df_sub in reader:
        output_address = (
                file_address
                + file_name[:-8]
                + "-"
                + str(index)
                + ".json.gz"
        )
        df_sub.to_json(
            output_address,
            orient="records",
            lines=True,
            compression="gzip"
        )
        index += 1
