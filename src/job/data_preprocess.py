# -*- coding: utf-8 -*-
#收集到的语料格式不同，需要统一格式；可能有重复的语句，导致测试集与训练集有交叉的部分，影响评估结果的有效性。
import sys
import os
path = os.getcwd()
path = os.path.dirname(path)
sys.path.append(path)

from model import data
import os, time
import hashlib
from algrithm.features import feature
from algrithm.indexs import invertedIndex
from model import data as data_tool
from config import load_models

data_set_names = ['crownpku_Small-Chinese-Corpus', 'icwb2-data', 'ProHiryu_bert-chinese-ner_data',
                  'renminribao_corpus', 'ner_data_LatticeLSTM', 'inews_data']

IINDEX_SENTENCE = {}
#读取crownpku_Small-Chinese-Corpus数据
def load_crownpku_Small_Chinese_Corpus():
    scr_dir = '../../data/corpus/crownpku_Small-Chinese-Corpus/'
    file_names = os.listdir(scr_dir)
    data_list = []
    for file_name in file_names:
        data_list += data.read_corpus_preprocess(scr_dir + file_name)
    #print("数据量是", len(data_list), data_list[:2])
    return data_list

def load_ProHiryu_bert_chinese_ner_data():
    scr_dir = '../../data/corpus/ProHiryu_bert-chinese-ner_data/'
    file_names = os.listdir(scr_dir)
    data_list = []
    for file_name in file_names:
        data_list += data.read_corpus_preprocess(scr_dir + file_name)
    #print("数据量是", len(data_list), data_list[:2])
    return data_list    

def load_ner_data_LatticeLSTM():
    '''这份数据的命名实体标记与常见的不同，需要替换一下:GPE-LOC, GEO-LOC(https://github.com/yanqiangmiffy/ner-english)'''
    scr_dir = '../../data/corpus/ner_data_LatticeLSTM/'
    file_names = os.listdir(scr_dir)
    data_list = []
    for file_name in file_names:
        data_list += data.read_corpus_preprocess(scr_dir + file_name)
    new_data_list = []
    for [chars, nertags, _] in data_list:
        for i in range(len(chars)):
            if nertags[i][-3:] in ['GPE', 'GEO']:
                nertags[i] = nertags[i][:2] + 'LOC'
            if nertags[i][0] =="M":
                nertags[i] = 'I' + nertags[i][1:]
        new_data_list.append([chars, nertags, scr_dir])
        
    #print("数据量是", len(data_list), new_data_list[:2])
    return new_data_list 
import re
def combine_words(line, chinese_family_name_set):
    comb_words = re.findall("\[.+?\]/.+?\s", line)
    for a_comb_word in comb_words:
#         print(a_comb_word)
        word, tag = a_comb_word.strip().split(']/')
        new_word = word[1:]
        word += ']'
        new_word = ''.join(map(lambda x: x.split('/')[0], new_word.split(' ')))
        line = line.replace(word, new_word)
    new_list_II = []
    word_tag_list = line.split()
    stack = []
    point = 0
    while point<len(word_tag_list):
        word_tag=word_tag_list[point]
        point += 1
        if word_tag[-2:]=='nr': 
            word, tag = word_tag.split('/')
            if word in chinese_family_name_set and len(stack)<=1:
                stack.append(word)
            else:
                stack.append(word)
                new_list_II.append("".join(stack) + "/nr")
                stack = []
        else:
            new_list_II.append(word_tag) 
    return new_list_II

def load_renminribao_corpus():

    postag2nertag = {'nr': 'PER', 'ns':"LOC", 'nt': "ORG", "nto": "ORG"}
    postag2other_tag = { 'mq': 'O', "nz": "O"}
    chinese_family_name = list(map(lambda x: x[0], open('../../data/named_entity_vocabs/FamilyName_300.txt', 'r', encoding='utf8').readlines()))
    chinese_family_name = list(map(lambda x: x.replace("\n", ''), chinese_family_name))
    chinese_family_name = set(chinese_family_name)
    print(chinese_family_name)
    
    '''人民日报的语料实际上是词性标注语料，需要把里面的人名，地名，机构名特别标注出来，并组织成所需的格式。'''
    #人民日报语料存在大量实体嵌套情况，以及由若干个词语组成一个实体的情况，需要专门处理；
    #中国人名一般会把姓和名分开，需要处理
    lines = list(open('../../data/corpus/renminribao_corpus/corpus.txt', 'r',  encoding='utf8'))
    lines = list(map(lambda x: x.split('/m  ')[1] if '/m  ' in x else "", lines))
    lines += list(open('../../data/corpus/renminribao_corpus/2014_corpus.txt', 'r',  encoding='utf8'))
    lines += list(open('../../data/corpus/renminribao_corpus/pro_corpus.txt', 'r',  encoding='utf8'))
    lines = list(filter(lambda x: len(x)>10, lines))
    lines = list(map(lambda x:x.replace("\n", ''), lines))
    print("人民日报语料的句子数量是", len(lines))
    data_list = []
    for line in lines:
        word_postags = combine_words(line, chinese_family_name)
        new_chars, new_postags = [], []
        for word_postag in word_postags:
            word_postag_list = word_postag.split('/')
            if len(word_postag_list)!=2:#有些词语内部有空格，数量是111句，暂时放弃这些词语
                continue
            [word, postag] = word_postag_list
            if postag not in postag2nertag or postag in postag2other_tag:
                for char in word: 
                    new_chars.append(char)
                    new_postags.append("O")
            else:
                ner_tag = postag2nertag[postag]
#                 print(word)
                new_chars.append(word[0])
                new_postags.append('B-' + ner_tag)
                for char in word[1:]:
                    new_chars.append(char)
                    new_postags.append('I-' + ner_tag)
        data_list.append([new_chars, new_postags, ''])
    print("人民日报数据量是", len(data_list), data_list[:2])
    return data_list
    
def load_inews_data():
    ner_tag_set = set({'PER', "LOC", "ORG"})
    data_path = '../../data/corpus/inews_data/ner_train.txt'
    lines = list(open(data_path, 'r', encoding='utf8'))
    lines = list(filter(lambda x: len(x)>10, lines))
    #print("inews的数据量是", len(lines))
    data_list = []
    for line in lines:
        words = line.replace("\n", '').split()
        char_list, ner_list = [], []
        for word in words:
            word_nertag_list = word.split('/')
            if len(word_nertag_list)==1:
                for char in word:
                    char_list.append(char)
                    ner_list.append("O")
            if len(word_nertag_list)==2 and len(word_nertag_list[0])> 0 and word_nertag_list[1] in ner_tag_set:
                ner_tag = word_nertag_list[1]
                word_part = word_nertag_list[0]
                if len(word_part)==1:
                    char_list.append(word_part)
                    ner_list.append('S-' + ner_tag)
                else:
                    char_list.append(word_part[0])
                    ner_list.append('B-' + ner_tag)
                    for char in word_part[1:-1]:
                        char_list.append(char)
                        ner_list.append('I-' + ner_tag)
                    char_list.append(word_part[-1])
                    ner_list.append('E-' + ner_tag)
        data_list.append([char_list, ner_list, data_path])
    return data_list
 
def load_weibo_data():
    ner_tag_set = set({'PER', "LOC", "ORG"})
    data_dir = '../../data/corpus/Weibo/'
    file_names = os.listdir(data_dir)
    lines = []
    for file_name in file_names:
        temp_lines = data.read_corpus_preprocess(data_dir + file_name)
        lines += temp_lines
    
    lines = list(map(lambda x: [list(map(lambda y: y[0], x[0])), x[1], x[2]], lines))
    #print(lines[0])
    for line in lines:
        for i in range(len(line[0])):
            if '.NAM' in line[1][i]:
                line[1][i] = line[1][i].replace('.NAM', '')
            else:
                line[1][i] = "O"
    new_data_list = []
    for [chars, nertags, _] in lines:
        for i in range(len(chars)):
            if nertags[i][-3:] in ['GPE', 'GEO']:
                nertags[i] = nertags[i][:2] + 'LOC'
        new_data_list.append([chars, nertags, data_dir])
    return lines 

# granularity is different
def load_Chinese_Literature_NER_RE_data():
    data_dir = '../../data/corpus./Chinese-Literature-NER-RE-Dataset/'
    file_names = os.listdir(data_dir)
    lines = []
    for file_name in file_names:
        temp_lines = data.read_corpus_preprocess(data_dir + file_name)
        lines += temp_lines
    
    #print(lines[0])
    for line in lines:
        for i in range(len(line[0])):
            if line[1][i][1:] == '_Person':
                line[1][i] = line[1][i].replace('_Person', '-PER')
            elif line[1][i][1:] == '_Location':
                line[1][i] = line[1][i].replace('_Location', '-LOC')
            elif line[1][i][1:] == '_Organization':
                line[1][i] = line[1][i].replace('_Organization', '-ORG')
            else:
                line[1][i] = "O"
    new_data_list = []
    for [chars, nertags, _] in lines:
        for i in range(len(chars)):
            if nertags[i][-3:] in ['GPE', 'GEO']:
                nertags[i] = nertags[i][:2] + 'LOC'
        new_data_list.append([chars, nertags, data_dir])
    #print("load_Chinese_Literature_NER_RE_data ", lines[:3])
    return lines 

def load_shiyybua_ner_data():
    data_dir = '../../data/corpus/shiyybua_ner'
    sentences = list(open('../../data/corpus/shiyybua_ner/source.txt', 'r', encoding='utf8').readlines())
    targets = list(open('../../data/corpus/shiyybua_ner/target.txt', 'r', encoding='utf8').readlines())
    data_list = []
    for i in range(len(sentences)):
        words = sentences[i].replace('\n', '').split()
        nertags = targets[i].replace('\n', '').split()
        temp_chars = []
        temp_nertags = []
        for j in range(len(words)):
            word = words[j]
            tag = nertags[j]
            if tag[-3:] in ['ORG', 'LOC', 'PER']:
                if tag[0] == 'B':
                    if len(word)==1:
                        temp_chars.append(word)
                        temp_nertags.append(tag)
                    else:
                        temp_chars += list(word)
                        temp_nertags += [tag] + ['I' + tag[1:]]*(len(word)-1)
                elif tag[0] == 'I':
                    if len(word)==1:
                        temp_chars.append(word)
                        temp_nertags.append(tag)
                    else:
                        temp_chars += list(word)
                        temp_nertags += [tag]*len(word)     
            else:
                temp_chars += list(word)
                temp_nertags += ['O']*len(word)
        data_list.append([temp_chars, temp_nertags, data_dir])
    #print(data_list[:2])
    return data_list
                      
def load_boson_data():

    scr_dir = '../../data/corpus/boson_data/boson_ner_format.txt'
    data_list = data.read_corpus_preprocess(scr_dir)
    new_data_list = []
    for [chars, nertags, _] in data_list:
        for i in range(len(chars)):
            #print(nertags[i])
            if nertags[i][-4:] not in ['-ORG', '-LOC', '-PER']:
                nertags[i] = "O"
        new_data_list.append([chars, nertags, scr_dir])
        
   # print("数据量是", len(data_list), new_data_list[:2])
    return new_data_list 

def load_ZR_Huang_ner_data():
    data_dir = '../../data/corpus/ZR_Huang_ner_data/'
    sentences = list(open(data_dir + 'source.txt', 'r', encoding='utf8'))[2:]
    nertags = list(open(data_dir + 'target.txt', 'r', encoding='utf8'))[2:]
    sentences = list(map(lambda x: x.replace('\n', '').split(' '), sentences))
    nertags = list(map(lambda x: x.replace('\n', '').split(' '), nertags))
    data_list = list(zip(sentences, nertags))
    data_list = list(map( lambda x: list(x) + [data_dir], data_list))
    data_list = list(filter(lambda x: len(x[0])==len(x[1]), data_list))
    #print(data_list[:5])
    return data_list

def load_mhcao916_ner_data():
    data_dir = '../../data/corpus/mhcao916_ner_data/'
    file_names = os.listdir(data_dir)
    lines = []
    for file_name in file_names:
        temp_lines = data.read_corpus_preprocess(data_dir + file_name)
        lines += temp_lines
    new_data_list = []
    for [chars, nertags, _] in lines:
        for i in range(len(chars)):
            #print(nertags[i])
            if nertags[i][-4:] in [ '-SCE', '-DLO']:
                nertags[i] = nertags[i][0] + '-LOC'
            elif nertags[i][-4:] in ['-SCE', '-HOT']:
                nertags[i] = nertags[i][0] + '-ORG'
            elif nertags[i][-4:] in ['-PER', '-ORG']:
                pass
            else:
                nertags[i] = "O"
        new_data_list.append([chars, nertags, data_dir])
    #print(new_data_list[:5])
    return new_data_list

def load_crownpku_ner_data():
    data_dir = '../../data/corpus/crownpku_ner_data/'
    file_names = os.listdir(data_dir)
    lines = []
    for file_name in file_names:
        temp_lines = data.read_corpus_preprocess(data_dir + file_name)
        lines += temp_lines
    return lines

def load_manu_labeled_data():
    ner_tag_set = set({'PER', "LOC", "ORG"})
    data_dir = '../../data/corpus/manu_labeled_data/'
    file_names = os.listdir(data_dir)
    lines = []
    for file_name in file_names:
        temp_lines = list(open(data_dir + file_name, 'r', encoding='utf8'))
        temp_lines = list(filter(lambda x: len(x)>10, temp_lines))
        lines += temp_lines
    #print("manu data 的数据量是", len(lines))
    data_list = []
    for line in lines:
        words = line.split()
        char_list, ner_list = [], []
        for word in words:
            word_nertag_list = word.split('/')
            if len(word_nertag_list)==1:
                for char in word:
                    char_list.append(char)
                    ner_list.append("O")
            if len(word_nertag_list)==2 and len(word_nertag_list[0])> 0 and word_nertag_list[1] in ner_tag_set:
                #print(word_nertag_list)
                char_list.append(word_nertag_list[0][0])
                ner_list.append("B-" + word_nertag_list[1])
                for char in word_nertag_list[0][1:]:
                    char_list.append(char)
                    ner_list.append('I-' + word_nertag_list[1])
        data_list.append([char_list, ner_list, data_dir])
    return data_list

def remove_dup(data_list):
    distinct_data_map = {}
    for sentence in data_list:
        sentence_str = ''.join(sentence[0]).replace(" ", '')
        ngrams = feature.get_ngrams(sentence_str, N=6, step=1)
        sparse = list(map(lambda x: [x, 0], ngrams))
        code = hashlib.md5(sentence_str.encode(encoding='utf_8')).hexdigest()
        
        same_sentences = invertedIndex.getSameSim(sparse, IINDEX_SENTENCE)
        invertedIndex.insert(code, sparse, IINDEX_SENTENCE)
        
        if len(same_sentences)>0: continue
        if code not in distinct_data_map and len(sentence[0])==len(sentence[1]):
            distinct_data_map[code] = sentence
        if len(distinct_data_map)%10000==0:
            print(len(distinct_data_map))
    
    print("原始数据的数量是", len(data_list),'去重后的数量是', len(distinct_data_map))
    data_list = list(distinct_data_map.values())
    return data_list

def load_normal(dir_path):
    file_names = os.listdir(dir_path)
    lines = []
    for file_name in file_names:
        temp_lines = data.read_corpus_preprocess(dir_path + file_name)
        lines += temp_lines
    return lines

def remove_spam(data_list):
    error_code_set = load_error_code()
    print(error_code_set)
    new_list = []
    i = 0
    for line in data_list:
        if if_all_O(line[1]) and random.uniform(0, 1)> 0.2: continue
        i+=1
        if i%10000==0:
            print("正在去除垃圾数据，", i)
        count = 0
        if len(line[0])<5:
            continue
        for char in line[0]:
#             print(char)
            if char in error_code_set:
#                 print(char, count, count/len(line[0]))
                count += 1
        if count/len(line[0])>0.5:
            print(line)
            continue
        else:
            new_list.append(line)
    return new_list
        
def remove_duplicate_sentences():
    data_list = []
#     data_list += load_crownpku_Small_Chinese_Corpus()
#     data_list += load_ProHiryu_bert_chinese_ner_data()
#     data_list += load_ner_data_LatticeLSTM()
#     data_list += load_renminribao_corpus()
#     data_list += load_weibo_data()#杂质较多
#     data_list += load_manu_labeled_data()
#     data_list += load_shiyybua_ner_data()
#     data_list += load_boson_data()
#     data_list += load_ZR_Huang_ner_data()
#     data_list += load_mhcao916_ner_data()
#     data_list += load_crownpku_ner_data()
#     data_list += load_normal("../../data/corpus/rmrb1997/")
    data_list += load_normal("../../data/corpus/ChineseNER/")
#     data_list += load_inews_data()
    ###data_list += load_Chinese_Literature_NER_RE_data()# granularity is different
    import random 
    random.shuffle(data_list)

    #split sentences. some lines contains multi sentences
    new_data_list = []
    for line in data_list:
        if len(line[0])!=len(line[1]): continue
        char_list, tags = [], []
        for i in range(len(line[0])):
            char_list.append(line[0][i])
            tags.append(line[1][i])
            if line[0][i] in ['。', '', '!'] :
                new_data_list.append([char_list, tags,  line[2]])
                char_list, tags = [], []
        if len(char_list)>0:
            new_data_list.append([char_list, tags, line[2]])
    data_list = remove_dup(new_data_list)
    data_list = remove_spam(data_list)
    data_list = data_list[:10000]
    print('data size after spliting sentences is ', len(data_list))   
    return data_list

def restore_data_as_crf_format(data_list, file_name):
    '''按照crf++的格式存储数据'''
    print("正在存储")
    with open('../../data/data_path/' + file_name, 'w', encoding='utf8') as f:
        for [chars, ner_tags] in data_list:
            for i in range(len(chars)):
                f.write(chars[i] + '\t' + ner_tags[i] + '\n')
            f.write('\n')
       
def change_label_to_BIESO(data_list):
    new_data_list = []
    for line in data_list:
#         print(line)
        [chars, nertags, _] = line
        nertags = list(map(lambda x: x.replace("M", "I"), nertags))
        nertags.append("O")
        chars.append("#")
        print(nertags)
        print(chars)
        print("#################")
        new_nertags, temp_nertags = [], []
        for i in range(len(chars)):
            if nertags[i][0] == "S":
                temp_nertags.append(nertags[i])
                new_nertags += temp_nertags
                temp_nertags = []
                continue
            if nertags[i][0] in ['O', "B"]:#如果当前tag是O
                if i-1>=0 and nertags[i-1][0] == 'B':
                    if i-2>0 and nertags[i-2][0] == 'B':
                        temp_nertags[-2] = 'S'+ nertags[i-2][1:]
                    print(i, chars[i],nertags[i][0], temp_nertags)
                    if len(temp_nertags)>0:
                        temp_nertags[-1] = 'S' + nertags[i-1][1:]
                    temp_nertags.append(nertags[i])
                    new_nertags += temp_nertags
                    temp_nertags = []
                elif i-1>=0 and nertags[i-1][0] == 'I':
                    #print(temp_nertags, i-1)
                    temp_nertags[-1] = 'E' + nertags[i-1][1:]
                    temp_nertags.append(nertags[i])
                    new_nertags += temp_nertags
                    temp_nertags = []
                else: 
                    temp_nertags.append(nertags[i])
                    new_nertags += temp_nertags
                    temp_nertags = []
            else:
                temp_nertags.append(nertags[i])
        new_nertags += temp_nertags
        #print(new_nertags)
        new_data_list.append([chars[:-1], new_nertags[:-1]])
    return new_data_list

def change_label_to_BIESO_new(data_list):
    new_data_list = []
    for line in data_list:
#         print(line)
        [chars_ori, nertags_ori, _] = line
        nertags_ori = list(map(lambda x: x.replace("M", "I"), nertags_ori))
        nertags, chars = nertags_ori[:], chars_ori[:]
        nertags.append("O")
        chars.append("#")
        my_stack = []
#         if_IOBES = False
        if nertags[0]!="O":
            my_stack.append(0)
        for i in range(1, len(chars)):
            if nertags[i][0] in ["B", "O", "S"]:
                if len(my_stack)==1:
                    nertags[my_stack[0]] = "S-" + nertags[my_stack[0]][2:]
                    my_stack = []
                    if nertags[i]!="O":
                        my_stack.append(i)
                elif len(my_stack)>1:
                    nertags[my_stack[-1]] = "E-" + nertags[my_stack[-1]][2:]
                    my_stack = []
                    if nertags[i]!="O":
                        my_stack.append(i)
                else:
                    if nertags[i]!="O":
                        my_stack.append(i)
            else:
                if nertags[i]!="O":
                    my_stack.append(i)
        #print(new_nertags)
#         print(" ".join(list(map(lambda x:x[0] + "/" +  x[1], zip(chars, nertags)))))
        new_data_list.append([chars[:-1], nertags[:-1]])
    return new_data_list

import random

def get_O_char_set(data_list):
    char_set = set({})
    for line in data_list:
        for i in range(len(line[0])):
            if line[1][i]=="O":
                char_set.add(line[0][i])
    return list(char_set)

import pickle
ENTITY_MAP = pickle.load(open("../../data/named_entity_vocabs/entity_map.pkl", 'rb'))
ORG_LIST = list(filter(lambda x: x[1]=="ORG", ENTITY_MAP.items()))
ORG_LIST = list(map(lambda x: x[0], ORG_LIST))
PER_LIST = list(filter(lambda x: x[1]=="PER", ENTITY_MAP.items()))
PER_LIST = list(map(lambda x: x[0], PER_LIST))
LOC_LIST = load_models.load_all_region_short()
ENTTY_NAME_MAP = {"ORG": ORG_LIST, "PER": PER_LIST, "LOC": LOC_LIST}
MAX_WORD_LEN = max(list(map(lambda x: len(x), ENTITY_MAP.keys())))

def add_ner_tag(new_tags, cand_word, tag):
    if len(cand_word)==1:new_tags.append("S-" + tag)
    if len(cand_word)>1:
        new_tags.append("B-" + tag)
        for _ in range(len(cand_word)-2): new_tags.append("I-" + tag)
        new_tags.append("E-" + tag) 
         
def add_outer_entity_label(chars, tags):
    text = "".join(chars)
    new_tags = []
    index = 0
    if_changed = False
    while index<len(text):
        if tags[index]!="O": 
            new_tags.append(tags[index])
            index += 1
            continue
        window_len = MAX_WORD_LEN
        #当索引快到文本右边界时，需要控制窗口长度，以免超出索引
        if index + MAX_WORD_LEN>=len(text): window_len = len(text)-index + 1
        if_word_here = False
        for j in range(window_len, 1, -1):#遍历这个窗口内的子字符串，查看是否有词表中的词语
            cand_word = text[index: index + j]
            if cand_word in ENTITY_MAP:
                tag = ENTITY_MAP[cand_word]
                add_ner_tag(new_tags, cand_word, tag)
                index += len(cand_word)
                if_word_here = True
                if_changed = True
                break
        if not if_word_here:
            new_tags.append(tags[index])
            index += 1
#         print(index, len(text), text)
    return new_tags, if_changed

def change_entity(chars, tags, entity_type):
    new_chars, new_tags = [], []
    i = 0
    if_changed = False
    changed_num=  0
    while i<len(chars):
        if tags[i][2:]==entity_type:
            if_changed = True
            this_entity_name = ''
            while i<len(chars) and tags[i][2:]==entity_type:
                this_entity_name += chars[i]
                i += 1
            if entity_type=="ORG":
                new_entity = ENTTY_NAME_MAP[entity_type][random.randint(0, len(ENTTY_NAME_MAP[entity_type])-1)]
#                 if "公司"  in this_entity_name and len(this_entity_name)>3:
# #                     print(this_entity_name)
#                     new_entity = ENTTY_NAME_MAP[entity_type][random.randint(0, len(ENTTY_NAME_MAP[entity_type])-1)]
#                     changed_num += 1
#                 else:
#                     new_entity =this_entity_name
            else:
                new_entity = ENTTY_NAME_MAP[entity_type][random.randint(0, len(ENTTY_NAME_MAP[entity_type])-1)]
            new_chars += list(new_entity)
            add_ner_tag(new_tags, new_entity, entity_type)
        else:
            new_chars.append(chars[i])
            new_tags.append(tags[i])
            i += 1
    if if_changed:
#         print("替换同类实体")
#         print("".join(chars))
#         print("".join(new_chars))
#         print("替换同类实体")
        return new_chars, new_tags
    else:
        return [], []

def if_all_O(tags):
    for tag in tags:
        if tag!="O": return False
    return True
   
def data_augment(data_list):
    new_data_list = []
    O_char_list = get_O_char_set(data_list)
    count = 0
    for line in data_list:
        new_chars, new_tags = [], []
        
        if random.uniform(0, 1)>0.99:
            flag = 0
            for i in range(len(line[0])):
                if line[0][i]=='O' and random.uniform(0,1) < 0.1: continue#随机删掉一个字
                new_chars.append(line[0][i])
                new_tags.append(line[1][i])
                flag = 1
            if flag==1: 
                new_data_list.append([new_chars, new_tags])
            flag = 0
            new_chars_I, new_tags_I = line[0][:], line[1][:]
            for i in range(len(line[0])):
                if line[0][i]=='O' and random.uniform(0,1) < 0.1:#随机替换一个字
                    new_chars_I[i] = O_char_list[random.randint(0, len(O_char_list)-1)]
                    flag = 1
            if flag==1: 
                new_data_list.append([new_chars_I, new_tags_I])
        
        #为漏标的企业名字打上ORG标签
#         try:
#             high_level_tags, if_changed = add_outer_entity_label(line[0], line[1])
#             if if_changed==True:
# #                 print("asdasdasd", line[0])
# #                 res = zip(line[0], high_level_tags)
# #                 print(res)
# #                 print("###########")
#                 new_data_list.append([line[0], high_level_tags])
#         except:
#             print("数据有问题")
            
        #开始替换同类实体
        #首先是组织
#         for _ in range(10):
#             if random.uniform(0, 1)>0.5: 
#                 new_chars, new_tags = change_entity(line[0], line[1], "ORG")
#             if random.uniform(0, 1)>0.9: 
#                 new_chars, new_tags = change_entity(new_chars, new_tags, "PER")
#             if random.uniform(0, 1)>0.9: 
#                 new_chars, new_tags = change_entity(new_chars, new_tags, "LOC")
#             if len(new_chars)>0 and new_chars!=line[0]:
#                 new_data_list.append([new_chars, new_tags])
        if if_tag_only(line[1], "ORG"):
            for _ in range(5):
                if random.uniform(0, 1)>0.5: 
                    new_chars, new_tags = change_entity(line[0], line[1], "ORG")
                    if len(new_chars)>0 and new_chars!=line[0]:
                        new_data_list.append([new_chars, new_tags])
        count += 1
#         print(count, len(data_list))
    print("开始")
    data_list += list(filter(lambda x: len(x[0])==len(x[1]), new_data_list))
    print("数据增强完毕")
    random.shuffle(data_list)
    return data_list

def if_tag_only(tags, tag):
    if_all_O = False
    for temp_tag in tags:
        if temp_tag!="O":
            if_all_O = True
        if temp_tag!="O" and temp_tag[-3:]!=tag:
            return False
    return True and if_all_O

def load_error_code():
    lines = list(open('../../data/corpus/error_chars.txt', 'r', encoding='utf8').readlines())
    error_code_freq = {}
    for line in lines:
        word = line.split("\t")
        if len(word)>0:
            error_code_freq[word[0]] = error_code_freq.get(word[0], 0) + 1
    error_code = sorted(error_code_freq.items(), key=lambda x: x[1], reverse=True)[:100]
    error_code_set = set({})
    for line in error_code:
        error_code_set.add(line[0])
    return error_code_set


from sklearn.model_selection import train_test_split
if __name__ == '__main__':
    
    data_list = remove_duplicate_sentences()
    print("开始修改标签")
    data_list = change_label_to_BIESO_new(data_list)
    print("开始分割训练集和测试集")
    train_data_list, test_data_list, _, _ = train_test_split(data_list, data_list, test_size=0.05)
    print("对训练集进行扩增")
    train_data_list = data_augment(train_data_list)
    print("存储数据")
    restore_data_as_crf_format(train_data_list, 'train_data')
    restore_data_as_crf_format(test_data_list, 'test_data')
# 
    print("开始构建word2id")
    data_tool.vocab_build('../../data/data_path/word2id.pkl', '../../data/data_path/train_data', min_count=1)
    print("开始抽取与训练的字向量")
    data_tool.extract_char_vec_from_pretrained()
 
     
     
    
