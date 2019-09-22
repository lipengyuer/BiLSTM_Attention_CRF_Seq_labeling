'''
Created on 2019年9月21日

@author: Administrator
'''
# 模型部分
import pickle
import os
from config import run_time
import numpy as np
import tensorflow as tf
from tensorflow.contrib.rnn import LSTMCell
from tensorflow.contrib.crf import crf_log_likelihood, crf_sequence_score
from tensorflow.contrib.crf import viterbi_decode
from utils.data_loader import pad_sequences, batch_yield
from utils import eval
import random
from utils import data_loader

## Session configuration
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # default: 0

class BiLSTMCRF():
    
    def __init__(self, 
                  use_pretrained_embeddings=True, embedding_dim=100, update_embedding=True,\
                  hidden_dim = 50, \
                  drop_out = 0.5
                 ):
        """
        mode: 制定训练模式或者预测模式
        embedding行：词向量相关配置
        """
        self.mode = 'train'
        print("开始初始化")
        self.word_id_map = pickle.load(open(run_time.PATH_WORD_ID, 'rb'))
        print("词汇表大小", len(self.word_id_map))
        self.id_word_map = {v: k for k, v in self.word_id_map.items()}
        self.init_embeddings(use_pretrained_embeddings, embedding_dim)
        self.build_graph(hidden_dim, drop_out)
    
    #初始化词向量
    def init_embeddings(self, use_pretrained_embeddings, embedding_dim): 
        print("初始化词向量。")
        if self.mode=='train':#如果是训练，加载预训练好的，或者随机初始化。
            if use_pretrained_embeddings==True:
                print("读取预训练的词向量")
                self.embeddings = pickle.load(open(run_time.PATH_PRETRAINED_EMBEDDINGS, 'rb'))
            else:
                print("随机初始化一份词向量")
#                 self.embeddings = 0.1*np.random.random((len(self.word_id_map), embedding_dim))
                self.embeddings = np.float32(np.random.uniform(-0.25, 0.25, \
                                                               (len(self.word_id_map), embedding_dim)))
        else:#如果是其他模式，加载模型自己训练得到的词向量即可
            print("加载模型自己的词向量")
            self.embeddings = pickle(open(run_time.PATH_EMBEDDINGS, 'rb')) 
               
    def build_graph(self, hidden_dim, drop_out, clip_grad=5.0):
        # 创建输入变量
        self.word_ids = tf.placeholder(tf.int32, shape=[None, None], \
                                       name="id_of_words")  # 文本切分后得到的单元id
        self.real_labels = tf.placeholder(tf.int32, shape=[None, None], name="real_labels")
        self.seq_length = tf.placeholder(tf.int32, shape=[None], name="length_of_sentence")
        self.lr = tf.placeholder(tf.float16, shape=[], name="learning_rate")  # 便于细致地训练模型
        
        # 将word转换为词向量
        with tf.variable_scope("words"):
            print(self.embeddings)
            self._word_embeddings = tf.Variable(self.embeddings, dtype=tf.float32, trainable=True,\
                                           name="_word_embeddings")#词向量是一个变量；当然也可以使用trainable冻结
            word_embeddings = tf.nn.embedding_lookup(params=self._word_embeddings, ids=self.word_ids,\
                                                     name="word_embeddings")#从词向量矩阵中，为词语找到对应的词向量，形成序列
        #词向量参数较多；由于语料等原因，噪声比较多。需要dropout,避免过拟合
        self.word_embeddings = tf.nn.dropout(word_embeddings, drop_out)
        
        #LSTM层
#         lstm_logits_0 = self.BiLSTM_layer(self.word_embeddings, self.seq_length, int(hidden_dim/2), layer_no=0)
        self.lstm_logits = self.BiLSTM_layer(self.word_embeddings, self.seq_length, hidden_dim, layer_no=1)
        self.loss = self.CRF_layer()
        
        #优化器
        optim = tf.train.AdamOptimizer(learning_rate=self.lr)
        grads_and_vars = optim.compute_gradients(self.loss)
        grads_and_vars_clip = [[tf.clip_by_value(g, -clip_grad, clip_grad), v] \
                                         for g, v in grads_and_vars]
        self.train_op = optim.apply_gradients(grads_and_vars_clip)
        
        #初始化图中的变量
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = 0.2  # need ~700MB GPU memory
        self.sess = tf.Session(config=config)
        self.sess.run(tf.global_variables_initializer())
                      
        #CRF层
    def CRF_layer(self):
        log_likelihood, self.transition_params = \
                   crf_log_likelihood(inputs=self.lstm_logits,tag_indices=self.real_labels,\
                                       sequence_lengths=self.seq_length)
        class_weight_data = tf.Variable(run_time.label_weight_array, dtype=tf.float32, trainable=False)
        self.ts_class_weight_ = tf.nn.embedding_lookup(params=class_weight_data, ids=self.real_labels)
        self.ts_class_weight = tf.reduce_mean(self.ts_class_weight_, axis=1)
        self.flag = tf.multiply(log_likelihood, self.ts_class_weight)
        loss = -tf.reduce_mean(tf.multiply(log_likelihood, self.ts_class_weight))
        loss = -tf.reduce_mean(log_likelihood)
        return loss
        
    def BiLSTM_layer(self, inputs, sequence_lengths, hidden_dim, layer_no=0, drop_out=0.5):        
        with tf.variable_scope("bilstm_" + str(layer_no)):
            cell_fw, cell_bw = LSTMCell(hidden_dim), LSTMCell(hidden_dim)
            (output_fw_seq, output_bw_seq), _ = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=cell_fw, cell_bw=cell_bw, inputs=inputs,
                sequence_length=sequence_lengths, dtype=tf.float32)
            output = tf.concat([output_fw_seq, output_bw_seq], axis=-1)
            output = tf.nn.dropout(output, drop_out)
            
        with tf.variable_scope("proj_" + str(layer_no)):
            print(run_time.TAG_NUM)
            self.W = tf.get_variable(name="W_" + str(layer_no),shape=[2 * hidden_dim, run_time.TAG_NUM],
                                initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)

            self.b = tf.get_variable(name="b_" + str(layer_no), shape=[run_time.TAG_NUM],
                                initializer=tf.zeros_initializer(), dtype=tf.float32)
        s = tf.shape(output)
        output = tf.reshape(output, [-1, 2*hidden_dim])
        pred = tf.matmul(output, self.W) + self.b
        logits = tf.reshape(pred, [-1, s[1], run_time.TAG_NUM])
        return logits
    
    def lr_decay(self, init_lr, step): return init_lr#init_lr*np.exp(-0.01*step**0.6)
    
    def fit(self, epoch_num=100, init_lr=0.01):
        self.mode = 'train'
        saver = tf.train.Saver(tf.global_variables())
        global_step = 0
        batch_size = 100
        for epoch in range(epoch_num):
            traing_data = data_loader.load_training_data(run_time.PATH_TRAINING_CORPUS, self.word_id_map\
                                                        , run_time.TAG_LABEL_MAP, batch_size=batch_size)
            testing_data = data_loader.load_training_data(run_time.PATH_TESTING_CORPUS, self.word_id_map\
                                                        , run_time.TAG_LABEL_MAP, batch_size=batch_size)
            for step, (seqs, labels) in enumerate(traing_data):
                global_step += 1
                current_lr = self.lr_decay(init_lr, global_step)
                data_dict, _ = self.get_feed_dict(seqs, labels=labels, \
                                lr=current_lr, dropout=0.5)
                _, loss_train = self.sess.run([self.train_op, self.loss], feed_dict=data_dict)
                if step%20==0:
                    print(epoch, step, "loss_train", loss_train, current_lr)
#                     print(te)
                if global_step%1000==0:
                    self.evaluate(testing_data)
        
                
    def get_feed_dict(self, seqs, labels=None, lr=None, dropout=None):
        """
        seqs:词语id序列，labels:标签id, lr:学习率
        :return: feed_dict
        """
        word_ids, seq_len_list = pad_sequences(seqs, pad_mark=0)
        feed_dict = {self.word_ids: word_ids, self.seq_length: seq_len_list}
        if labels is not None:
            labels_, _ = pad_sequences(labels, pad_mark=0)
            feed_dict[self.real_labels] = labels_
        if lr is not None: feed_dict[self.lr] = lr
        return feed_dict, seq_len_list

    def predict(self):
        pass
       
    def evaluate(self, testing_data):
        all_pred_tags, all_real_tags, all_chars = [], [], []
        for step, (seq_list, labels_list) in enumerate(testing_data):
            data_dict, seq_len_list = self.get_feed_dict(seq_list, labels=labels_list, dropout=1.0)
            logits, transition_params = self.sess.run([self.lstm_logits, self.transition_params],
                                             feed_dict=data_dict)
            for logit, seq_len, labels, seq in zip(logits, seq_len_list, labels_list, seq_list):
                viterbi_seq, _ = viterbi_decode(logit[:seq_len], transition_params)
                pred_tag = [run_time.LABEL_TAG_MAP[label] for label in viterbi_seq]
                real_tags = [run_time.LABEL_TAG_MAP[label] for label in labels]
                chars = [self.id_word_map[label] for label in seq]
                all_pred_tags += pred_tag
                all_real_tags += real_tags
                all_chars += chars
        lines = [all_chars[i] + ' ' + all_real_tags[i] + ' ' + all_pred_tags[i] + '\n' \
                               for i in range(len(all_pred_tags))]
        test_result_path = '../../data/output/test_label.txt'
        test_metric_path = '../../data/output/metric.txt'
        with open(test_result_path, 'w', encoding='utf8') as f:
            f.writelines(lines)
        test_result_metric = eval.conlleval(test_result_path, test_metric_path)
        test_result_metric = '\n'.join(test_result_metric)
        print(test_result_metric)
    
#     def class_weight_I(self):
        #         error_score = 0
#         #为各个类别加权，以减小不均衡的影响
#         lstm_logits_array = lstm_logits.eval()
#         real_labels_array = real_labels.eval()
#         sequence_length_array = sequence_lengths.eval()
#         count_map = {'PER': [0, 0], "ORG": [0, 0],"LOC": [0, 0]}
#         for i in range(lstm_logits.shape[0]):
#             logit, seq_len = lstm_logits_array[i], sequence_length_array[i]
#             viterbi_seq, _ = viterbi_decode(logit[:seq_len], self.transition_params)
#             temp_real_labels = real_labels_array[i]
#             for j in range(len(viterbi_seq)):
#                 real_tag = run_time.LABEL_TAG_MAP[temp_real_labels[j]]
#                 count_map[real_tag][0] += 1
#                 if temp_real_labels[j]!=viterbi_seq[j]:
#                     count_map[real_tag][1] += 1
#         error = {k: v[1]/v[0] for k, v in count_map.items()}
#         for k, v in error: error_score += v*run_time.class_weight_map[k]
#         class_weight_data = tf.Variable(run_time.label_weight_array, dtype=tf.float32)
#         ts_class_weight = tf.nn.embedding_lookup(params=class_weight_data, ids=self.real_labels)

