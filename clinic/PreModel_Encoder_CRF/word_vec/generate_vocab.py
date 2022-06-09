# /usr/bin/env python
# coding=utf-8
import json
import jieba
from pathlib import Path
from collections import Counter


def generate_voc(freq=2):
    """利用分词工具生成词向量词表
    """
    # load dict
    jieba.load_userdict('/home/zcm/projs/ccks_ner/clinic/PreModel_Encoder_CRF/word_vec/custom_vocab.txt')
    # get all data
    sentences = []
    data_dir = ['/home/zcm/projs/ccks_ner/clinic/PreModel_Encoder_CRF/data/train/', '/home/zcm/projs/ccks_ner/clinic/PreModel_Encoder_CRF/data/val/', '/home/zcm/projs/ccks_ner/clinic/PreModel_Encoder_CRF/data/test/']
    for dir_ in data_dir:
        with open(dir_ + 'sentences.txt', 'r', encoding='utf-8') as f_sen:
            sentences.extend([''.join(line.strip().split(' ')) for line in f_sen])
    # 分词
    all_words = []
    for sen in sentences:
        all_words.extend([w for w in jieba.lcut(sen, cut_all=False) if len(w) > 1])

    with open('vocab.txt', 'w', encoding='utf-8') as f_voc:
        for w in Counter(all_words).most_common():
            if w[1] >= freq:
                f_voc.write(f'{w[0]}\t{w[1]}\n')


if __name__ == '__main__':
    generate_voc(freq=10)
