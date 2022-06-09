#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""dataloader.py utils"""
import re
import random
import math
import os
from cblue.utils import load_json, load_dict, write_dict
from torch.utils.data import Dataset

class EEDataProcessor(object):
    def __init__(self, root, is_lower=True, no_entity_label='O'):
        self.task_data_dir = os.path.join(root, 'CMeEE')
        self.train_path = os.path.join(self.task_data_dir, 'CMeEE_train.json')
        self.dev_path = os.path.join(self.task_data_dir, 'CMeEE_dev.json')
        self.test_path = os.path.join(self.task_data_dir, 'CMeEE_test.json')

        self.label_map_cache_path = os.path.join(self.task_data_dir, 'CMeEE_label_map.dict')
        self.label2id = None
        self.id2label = None
        self.no_entity_label = no_entity_label
        self._get_labels()
        self.num_labels = len(self.label2id.keys())

        self.is_lower = is_lower

    def get_train_sample(self):
        return self._pre_process(self.train_path, is_predict=False)

    def get_dev_sample(self):
        return self._pre_process(self.dev_path, is_predict=False)

    def get_test_sample(self):
        return self._pre_process(self.test_path, is_predict=True)

    def _get_labels(self):
        if os.path.exists(self.label_map_cache_path):
            label_map = load_dict(self.label_map_cache_path)
        else:
            label_set = set()
            samples = load_json(self.train_path)
            for sample in samples:
                for entity in sample["entities"]:
                    label_set.add(entity['type'])
            label_set = sorted(label_set)
            labels = [self.no_entity_label]
            for label in label_set:
                labels.extend(["B-{}".format(label), "I-{}".format(label)])
            label_map = {idx: label for idx, label in enumerate(labels)}
            write_dict(self.label_map_cache_path, label_map)
        self.id2label = label_map
        self.label2id = {val: key for key, val in self.id2label.items()}

    def _pre_process(self, path, is_predict):
        def label_data(data, start, end, _type):
            """label_data"""
            for i in range(start, end + 1):
                suffix = "B-" if i == start else "I-"
                data[i] = "{}{}".format(suffix, _type)
            return data

        outputs = {'text': [], 'label': [], 'orig_text': []}
        samples = load_json(path)
        for data in samples:
            if self.is_lower:
                text_a = ["，" if t == " " or t == "\n" or t == "\t" else t
                          for t in list(data["text"].lower())]
            else:
                text_a = ["，" if t == " " or t == "\n" or t == "\t" else t
                          for t in list(data["text"])]
            # text_a = "\002".join(text_a)
            outputs['text'].append(text_a)
            outputs['orig_text'].append(data['text'])
            if not is_predict:
                labels = [self.no_entity_label] * len(text_a)
                for entity in data['entities']:
                    start_idx, end_idx, type = entity['start_idx'], entity['end_idx'], entity['type']
                    labels = label_data(labels, start_idx, end_idx, type)
                outputs['label'].append('\002'.join(labels))
            elif is_predict:
                labels = [self.no_entity_label] * len(text_a)
                outputs['label'].append('\002'.join(labels))
        return outputs

    def extract_result(self, results, test_input):
        predicts = []
        for j in range(len(results)):
            text = "".join(test_input[j])
            ret = []
            entity_name = ""
            flag = []
            visit = False
            start_idx, end_idx = 0, 0
            for i, (char, tag) in enumerate(zip(text, results[j])):
                tag = self.id2label[tag]
                if tag[0] == "B":
                    if entity_name != "":
                        x = dict((a, flag.count(a)) for a in flag)
                        y = [k for k, v in x.items() if max(x.values()) == v]
                        ret.append({"start_idx": start_idx, "end_idx": end_idx, "type": y[0], "entity": entity_name})
                        flag.clear()
                        entity_name = ""
                    visit = True
                    start_idx = i
                    entity_name += char
                    flag.append(tag[2:])
                    end_idx = i
                elif tag[0] == "I" and visit:
                    entity_name += char
                    flag.append(tag[2:])
                    end_idx = i
                else:
                    if entity_name != "":
                        x = dict((a, flag.count(a)) for a in flag)
                        y = [k for k, v in x.items() if max(x.values()) == v]
                        ret.append({"start_idx": start_idx, "end_idx": end_idx, "type": y[0], "entity": entity_name})
                        flag.clear()
                    start_idx = i + 1
                    visit = False
                    flag.clear()
                    entity_name = ""

            if entity_name != "":
                x = dict((a, flag.count(a)) for a in flag)
                y = [k for k, v in x.items() if max(x.values()) == v]
                ret.append({"start_idx": start_idx, "end_idx": end_idx, "type": y[0], "entity": entity_name})
            predicts.append(ret)
        return predicts

class EEDataset(Dataset):
    def __init__(
            self,
            samples,
            data_processor,
            mode='train',
            max_length=128,
            model_type='bert',
            ngram_dict=None
    ):
        super(EEDataset, self).__init__()

        self.orig_text = samples['orig_text']
        self.texts = samples['text']

        self.labels = samples['label']

        self.data_processor = data_processor
        self.max_length = max_length
        self.mode = mode
        self.ngram_dict = ngram_dict
        self.model_type = model_type

    def __len__(self):
        return len(self.texts)


def split_text(text, max_len, split_pat=r'([，。]”?)', greedy=False):
    """文本分片
    将超过长度的文本分片成多段满足最大长度要求的最长连续子文本
    约束条件：1）每个子文本最大长度不超过max_len；
             2）所有的子文本的合集要能覆盖原始文本。
    Arguments:
        text {str} -- 原始文本
        max_len {int} -- 最大长度

    Keyword Arguments:
        split_pat {str or re pattern} -- 分割符模式 (default: {SPLIT_PAT})
        greedy {bool} -- 是否选择贪婪模式 (default: {False})
                         贪婪模式：在满足约束条件下，选择子文本最多的分割方式
                         非贪婪模式：在满足约束条件下，选择冗余度最小且交叉最为均匀的分割方式

    Returns:
        tuple -- 返回子文本列表以及每个子文本在原始文本中对应的起始位置列表

    Examples:
        text = '今夕何夕兮，搴舟中流。今日何日兮，得与王子同舟。蒙羞被好兮，不訾诟耻。心几烦而不绝兮，得知王子。山有木兮木有枝，心悦君兮君不知。'
        sub_texts, starts = split_text(text, maxlen=30, greedy=False)
        for sub_text in sub_texts:
            print(sub_text)
        print(starts)
        for start, sub_text in zip(starts, sub_texts):
            if text[start: start + len(sub_text)] != sub_text:
            print('Start indice is wrong!')
            break
    """
    # 文本小于max_len则不分割
    if len(text) <= max_len:
        return [text], [0]
    # 分割字符串
    segs = re.split(split_pat, text)
    # init
    sentences = []
    # 将分割后的段落和分隔符组合
    for i in range(0, len(segs) - 1, 2):
        sentences.append(segs[i] + segs[i + 1])
    if segs[-1]:
        sentences.append(segs[-1])
    n_sentences = len(sentences)
    sent_lens = [len(s) for s in sentences]

    # 所有满足约束条件的最长子片段
    alls = []
    for i in range(n_sentences):
        length = 0
        sub = []
        for j in range(i, n_sentences):
            if length + sent_lens[j] <= max_len or not sub:
                sub.append(j)
                length += sent_lens[j]
            else:
                break
        alls.append(sub)
        # 将最后一个段落加入
        if j == n_sentences - 1:
            if sub[-1] != j:
                alls.append(sub[1:] + [j])
            break

    if len(alls) == 1:
        return [text], [0]

    if greedy:
        # 贪婪模式返回所有子文本
        sub_texts = [''.join([sentences[i] for i in sub]) for sub in alls]
        starts = [0] + [sum(sent_lens[:i]) for i in range(1, len(alls))]
        return sub_texts, starts
    else:
        # 用动态规划求解满足要求的最优子片段集
        DG = {}  # 有向图
        N = len(alls)
        for k in range(N):
            tmplist = list(range(k + 1, min(alls[k][-1] + 1, N)))
            if not tmplist:
                tmplist.append(k + 1)
            DG[k] = tmplist

        routes = {}
        routes[N] = (0, -1)
        for i in range(N - 1, -1, -1):
            templist = []
            for j in DG[i]:
                cross = set(alls[i]) & (set(alls[j]) if j < len(alls) else set())
                w_ij = sum([sent_lens[k] for k in cross]) ** 2  # 第i个节点与第j个节点交叉度
                w_j = routes[j][0]  # 第j个子问题的值
                w_i_ = w_ij + w_j
                templist.append((w_i_, j))
            routes[i] = min(templist)

        sub_texts, starts = [''.join([sentences[i] for i in alls[0]])], [0]
        k = 0
        while True:
            k = routes[k][1]
            sub_texts.append(''.join([sentences[i] for i in alls[k]]))
            starts.append(sum(sent_lens[: alls[k][0]]))
            if k == N - 1:
                break

    return sub_texts, starts


class InputExample(object):
    """a single set of samples of data_src
    """

    def __init__(self, sentence, tag):
        self.sentence = sentence
        self.tag = tag


class InputFeatures(object):
    """
    Desc:
        a single set of features of data_src
    """

    def __init__(self,
                 input_ids,
                 input_mask,
                 tag,
                 split_to_original_id,
                 example_id
                 ):
        self.input_mask = input_mask
        self.input_ids = input_ids
        self.tag = tag

        # use to split
        self.split_to_original_id = split_to_original_id
        self.example_id = example_id


def read_examples(data_dir, data_sign):
    """read data_src to InputExamples
    :return examples (List[InputExample])
    """
    examples = []

    data_processor = EEDataProcessor(root=data_dir, is_lower=False)
    dataset_class = EEDataset
    if data_sign == 'train':
        train_samples = data_processor.get_train_sample()
        train_dataset = dataset_class(train_samples, data_processor, mode='train',
                                      model_type='hmm', ngram_dict=None, max_length=128)
        for label_idx in range(len(train_dataset.labels)):  # 15000
            train_dataset.labels[label_idx] = train_dataset.labels[label_idx].split('\002')
        word_lists = train_dataset.texts
        tag_lists = train_dataset.labels

    elif data_sign == 'val':
        eval_samples = data_processor.get_dev_sample()
        eval_dataset = dataset_class(eval_samples, data_processor, mode='eval',
                                     model_type='hmm', ngram_dict=None, max_length=128)
        for label_idx in range(len(eval_dataset.labels)):
            eval_dataset.labels[label_idx] = eval_dataset.labels[label_idx].split('\002')
        word_lists = eval_dataset.texts
        tag_lists = eval_dataset.labels

    elif data_sign == 'test':
        test_samples = data_processor.get_test_sample()
        test_dataset = dataset_class(test_samples, data_processor, mode='test', ngram_dict=None,
                                     max_length=128, model_type='hmm')
        for label_idx in range(len(test_dataset.labels)):
            test_dataset.labels[label_idx] = test_dataset.labels[label_idx].split('\002')
        word_lists = test_dataset.texts
        tag_lists = test_dataset.labels

    for sen, tag in zip(word_lists, tag_lists):
        example = InputExample(sentence=sen, tag=tag)
        examples.append(example)    

    # read src data
    '''
    with open(data_dir / f'{data_sign}/sentences.txt', "r", encoding='utf-8') as f_sen, \
            open(data_dir / f'{data_sign}/tags.txt', 'r', encoding='utf-8') as f_tag:
        for sen, tag in zip(f_sen, f_tag):
            example = InputExample(sentence=sen.strip().split(' '), tag=tag.strip().split(' '))
            examples.append(example)
    print("InputExamples:", len(examples))
    '''
    return examples


#def convert_examples_to_features(params, examples, tokenizer, pad_sign=True, greed_split=True):
def convert_examples_to_features(params, examples, tokenizer, word_dict, pad_sign=True, greed_split=True):
    """convert examples to features.
    :param examples (List[InputExamples])
    :param pad_sign: 是否补零
    """
    # tag to id
    tag2idx = {tag: idx for idx, tag in enumerate(params.tags)}
    features = []

    # context max len
    max_len = params.max_seq_length
    split_pad = r'([,.!?，。！？]”?)'

    for (example_idx, example) in enumerate(examples):
        # split long text
        sub_texts, starts = split_text(text=''.join(example.sentence), max_len=max_len,
                                       greedy=greed_split, split_pat=split_pad)
        original_id = list(range(len(example.sentence)))

        # get split features
        for text, start in zip(sub_texts, starts):
            # tokenize返回为空则设为[UNK]
            text_tokens = [tokenizer.tokenize(token)[0] if len(tokenizer.tokenize(token)) == 1 else '[UNK]'
                           for token in text]
            # label id
            tag_ids = [tag2idx[tag] for tag in example.tag[start:start + len(text)]]
            # 保存子文本对应原文本的位置
            split_to_original_id = original_id[start:start + len(text)]

            assert len(tag_ids) == len(split_to_original_id), 'check the length of tag_ids and split_to_original_id!'

            # cut off
            if len(text_tokens) > max_len:
                text_tokens = text_tokens[:max_len]
                tag_ids = tag_ids[:max_len]
                split_to_original_id = split_to_original_id[:max_len]
            # token to id
            text_ids = tokenizer.convert_tokens_to_ids(text_tokens)

            # sanity check
            assert len(text_ids) == len(tag_ids), f'check the length of text_ids and tag_ids!'
            assert len(text_ids) == len(split_to_original_id), f'check the length of text_ids and split_to_original_id!'

            # zero-padding up to the sequence length
            if len(text_ids) < max_len and pad_sign:
                # 补零
                pad_len = max_len - len(text_ids)
                # token_pad_id=0
                text_ids += [0] * pad_len
                tag_ids += [tag2idx['O']] * pad_len
                split_to_original_id += [-1] * pad_len

            # mask
            input_mask = [1 if idx > 0 else 0 for idx in text_ids]

            word_matches = []
            #  Filter the word segment from 2 to 7 to check whether there is a word
            for p in range(2, 8):
                for q in range(0, len(text_tokens) - p + 1):
                    character_segment = text_tokens[q:q + p]
                    # j is the starting position of the word
                    # i is the length of the current word
                    character_segment = tuple(character_segment)
                    if character_segment in word_dict.word_to_id_dict:
                        word_index = word_dict.word_to_id_dict[character_segment]
                        word_matches.append([word_index, q, p, character_segment])

            random.shuffle(word_matches)
            # max_word_in_seq_proportion = max_word_in_seq
            max_word_in_seq_proportion = math.ceil((len(text_tokens) / max_len) * word_dict.max_word_in_seq)
            if len(word_matches) > max_word_in_seq_proportion:
                word_matches = word_matches[:max_word_in_seq_proportion]

            word_ids = [word[0] for word in word_matches]
            word_positions = [word[1] for word in word_matches]
            word_lengths = [word[2] for word in word_matches]
            word_tuples = [word[3] for word in word_matches]

            import numpy as np

            # record the masked positions
            word_positions_matrix = np.zeros(shape=(max_len, word_dict.max_word_in_seq), dtype=np.int32)
            for i in range(len(word_ids)):
                word_positions_matrix[word_positions[i]:word_positions[i] + word_lengths[i], i] = 1.0

            # Zero-pad up to the max word in seq length.
            padding = [0] * (word_dict.max_word_in_seq - len(word_ids))
            word_ids += padding

            # get features
            features.append(
                InputFeatures(
                    input_ids=text_ids,
                    tag=tag_ids,
                    input_mask=input_mask,
                    split_to_original_id=split_to_original_id,
                    example_id=example_idx
                ))

    return features
