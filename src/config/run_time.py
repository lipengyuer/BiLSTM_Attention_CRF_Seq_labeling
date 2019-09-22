import numpy as np
# 运行过程中的参数
LSTM_LAYER_NUM = 1  # BiLSTM的层数
HIDDEN_DIM = 100  # 一层LSTM的神经元个数，默认每层都一样大
TAG_LABEL_MAP = {"O": 0, "S-PER": 1, "B-PER": 2, "I-PER": 3, "E-PER": 4,
             "B-LOC": 5, "I-LOC": 6, "E-LOC": 7, "S-LOC": 11,
             "B-ORG": 8, "I-ORG": 9, "E-ORG": 10, "S-ORG": 12,
             }  # 标签及其label对应关系
LABEL_TAG_MAP =  {v: k for k, v in TAG_LABEL_MAP.items()}
TAG_NUM = len(TAG_LABEL_MAP)#标签的种类个数
class_weight_map = {"ORG": 0.2, "PER": 0.5, "LOC": 1.0}
label_weight_map = {}
for label, tag in LABEL_TAG_MAP.items():
    label_weight_map[label] = class_weight_map.get(tag[-3:], 1.0)
label_weight_map[0] = 1.0
label_weight_array = np.zeros(len(TAG_LABEL_MAP))
for k, v in label_weight_map.items(): label_weight_array[k]  = v
# print(label_weight_array)

PATH_WORD_ID = "../../data/data_path/word2id.pkl"
PATH_PRETRAINED_EMBEDDINGS = "../../data/data_path/pretrained_char_vec.pkl"
PATH_EMBEDDINGS = "../../data/data_path/embeddings.pkl"
DIR_CHECK_POINT = "../../data/data_path/check_points"

PATH_TRAINING_CORPUS = "../../data/data_path/train_data"
PATH_TESTING_CORPUS = "../../data/data_path/test_data"
