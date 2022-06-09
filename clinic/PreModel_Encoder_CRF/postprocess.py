# /usr/bin/env python
# coding=utf-8
import os
from cgi import test
import json
import argparse

import pandas as pd
from dataloader import NERDataLoader
from dataloader_utils import read_examples

from utils import Params, IO2STR
from metrics import get_entities

# 参数解析器
parser = argparse.ArgumentParser()
parser.add_argument('--ex_index', type=int, default=2, help="实验名称索引")
parser.add_argument('--mode', type=str, default='test', help="后处理结果类型")


def pred_res(orig_text, test_res, test_word_lists, result_output_dir, tag2id, id2label):
    predictions = []
    test_inputs = test_word_lists
    for tes in test_res:
        predictions.append(word_trans_tag(tes, tag2id))
    # predictions = [pred[1:-1] for pred in predictions]
    predicts = extract_result(predictions, test_inputs, id2label)
    ee_commit_prediction(orig_text=orig_text, preds=predicts, output_dir=result_output_dir)

def word_trans_tag(words, tag2id):
    res = []
    for w in words:
        res.append(tag2id[w])
    return res    

def extract_result(results, test_input, id2label):
    predicts = []
    for j in range(len(results)):
        text = "".join(test_input[j])
        ret = []
        entity_name = ""
        flag = []
        visit = False
        start_idx, end_idx = 0, 0
        for i, (char, tag) in enumerate(zip(text, results[j])):
            tag = id2label[tag]
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

def ee_commit_prediction(orig_text, preds, output_dir):
    pred_result = []
    for item in zip(orig_text, preds):
        tmp_dict = {'text': item[0], 'entities': item[1]}
        pred_result.append(tmp_dict)
    with open(os.path.join(output_dir, 'CMeEE_test.json'), 'w', encoding='utf-8') as f:
        f.write(json.dumps(pred_result, indent=2, ensure_ascii=False))


def apply_fn(group):
    result = []
    # 获取该组的所有实体
    for tags, s2o in zip(group.tags, group.split_to_ori):
        entities = get_entities(eval(tags))
        for entity in entities:
            result.append((entity[0], eval(s2o)[entity[1]], eval(s2o)[entity[2]]))
    return result

def load_dict(dict_path):
    """load_dict"""
    vocab = {}
    for line in open(dict_path, 'r', encoding='utf-8'):
        key, value = line.strip('\n').split('\t')
        vocab[int(key)] = value
    return vocab

def postprocess(params, mode):
    id2label = load_dict("/home/zcm/projs/CBLUE/CBLUEDatasets/CMeEE/CMeEE_label_map.dict")
    tag2id = {val: key for key, val in id2label.items()}
    '''
    dataloader = NERDataLoader(params)
    loader = dataloader.get_dataloader(data_sign=mode)
    data_text = 
    # get text
    with open('/home/zcm/projs/ccks_ner/clinic/PreModel_Encoder_CRF/ccks2_task1_val/ccks2020_2_task1_test_set_no_answer.txt', 'r', encoding='utf-8') as f_scr_test, \
            open('/home/zcm/projs/ccks_ner/clinic/PreModel_Encoder_CRF/data/test/sentences.txt', 'r', encoding='utf-8') as f_src_val:
        if mode == 'test':
            data_text = [dict(eval(line.strip()))["originalText"] for line in f_scr_test]
        else:
            data_text = [''.join(line.strip().split(' ')) for line in f_src_val]
    '''
    # with open(params.params_path / f'submit_{mode}.txt', 'w', encoding='utf-8') as f_sub:
        # get df
    pre_df = pd.read_csv(params.params_path / f'{mode}_tags_pre.csv', encoding='utf-8')
    pre_df = pd.DataFrame(pre_df.groupby('example_id').apply(apply_fn), columns=['entities']).reset_index()

    examples = read_examples('/home/zcm/projs/ccks_ner/clinic/PreModel_Encoder_CRF/data/CBLUEDatasets', data_sign='test')
    test_res = []
    orig_text = []
    # write to json file
    for entities, example in zip(pre_df['entities'], examples):
        # sample_list = []
        
        for entity in set(entities):
            example.tag[entity[1]] = "{}{}".format("B-", entity[0].strip())

            # enti_dict = {}
            # enti_dict["label_type"] = IO2STR[entity[0].strip()]
            # enti_dict["start_pos"] = entity[1]
            # enti_dict["end_pos"] = entity[2] + 1
            for i in range(entity[1] + 1, entity[2] + 1):
                example.tag[i] = "{}{}".format("I-", entity[0].strip())            
            
            # sample_list.append(enti_dict)
        sen = "".join(example.sentence)
        orig_text.append(sen)
        test_res.append(example.tag)
        
    pred_res(orig_text=orig_text, test_res=test_res, test_word_lists =orig_text, 
            result_output_dir="/home/zcm/projs/ccks_ner/clinic/PreModel_Encoder_CRF/data/test/", tag2id=tag2id, id2label=id2label)

def pred_res(orig_text, test_res, test_word_lists, result_output_dir, tag2id, id2label):
    predictions = []
    test_inputs = test_word_lists
    for tes in test_res:
        predictions.append(word_trans_tag(tes, tag2id))
    # predictions = [pred[1:-1] for pred in predictions]
    predicts = extract_result(predictions, test_inputs, id2label)
    ee_commit_prediction(orig_text=orig_text, preds=predicts, output_dir=result_output_dir)

def word_trans_tag(words, tag2id):
    res = []
    for w in words:
        res.append(tag2id[w])
    return res    

def extract_result(results, test_input, id2label):
    predicts = []
    for j in range(len(results)):
        text = "".join(test_input[j])
        ret = []
        entity_name = ""
        flag = []
        visit = False
        start_idx, end_idx = 0, 0
        for i, (char, tag) in enumerate(zip(text, results[j])):
            tag = id2label[tag]
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

def ee_commit_prediction(orig_text, preds, output_dir):
    pred_result = []
    for item in zip(orig_text, preds):
        tmp_dict = {'text': item[0], 'entities': item[1]}
        pred_result.append(tmp_dict)
    with open(os.path.join(output_dir, 'CMeEE_test.json'), 'w', encoding='utf-8') as f:
        f.write(json.dumps(pred_result, indent=2, ensure_ascii=False))

if __name__ == '__main__':
    args = parser.parse_args()
    params = Params(ex_index=args.ex_index)
    postprocess(params, mode=args.mode)
