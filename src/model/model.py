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


class BiLSTMCRF():
    
    def __init__(self, 
                  use_pretrained_embeddings=True, embedding_dim=100, update_embedding=True,\
                  hidden_dim = [150], \
                  if_multi_layer=0
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
        self.build_graph(hidden_dim=hidden_dim, if_multi_layer=if_multi_layer)
    
    #初始化词向量
    def init_embeddings(self, use_pretrained_embeddings, embedding_dim): 
        print("初始化词向量。")
        if self.mode=='train':#如果是训练，加载预训练好的，或者随机初始化。
            if use_pretrained_embeddings==True:
                print("读取预训练的词向量")
                self.embeddings = pickle.load(open(run_time.PATH_PRETRAINED_EMBEDDINGS, 'rb'))
            else:
                print("随机初始化一份词向量")
                self.embeddings = np.float32(np.random.uniform(-0.25, 0.25, \
                                                               (len(self.word_id_map), embedding_dim)))
        else:#如果是其他模式，加载模型自己训练得到的词向量即可
            print("加载模型自己的词向量")
            self.embeddings = pickle(open(run_time.PATH_EMBEDDINGS, 'rb')) 
        print("词向量shape", self.embeddings)
        # 将word转换为词向量
        with tf.variable_scope("words"):
            print(self.embeddings)
            self._word_embeddings = tf.Variable(self.embeddings, dtype=tf.float32, trainable=True,\
                                           name="_word_embeddings")#词向量是一个变量；当然也可以使用trainable冻结
            
    def build_graph(self, clip_grad=5.0, hidden_dim=[150], if_multi_layer=0):
        # 创建输入变量
        self.word_ids = tf.placeholder(tf.int32, shape=[None, None], \
                                       name="id_of_words")  # 文本切分后得到的单元id
        
        self.real_labels = tf.placeholder(tf.int32, shape=[None, None], name="real_labels")
        self.seq_length = tf.placeholder(tf.int32, shape=[None], name="length_of_sentence")
        self.lr = tf.placeholder(tf.float16, shape=[], name="learning_rate")  # 便于细致地训练模型
        self.batch_size = tf.placeholder(tf.int32, shape=[], name="batch_size")  # 便于细致地训练模型
        self.dropout =  tf.placeholder(dtype=tf.float32, shape=[], name="dropout")

        word_embeddings = tf.nn.embedding_lookup(params=self._word_embeddings, ids=self.word_ids,\
                                                 name="word_embeddings")#从词向量矩阵中，为词语找到对应的词向量，形成序列
        #词向量参数较多；由于语料等原因，噪声比较多。需要dropout,避免过拟合
        self.word_embeddings = tf.nn.dropout(word_embeddings, self.dropout)
        
        #LSTM层
        if if_multi_layer==1:
            self.multi_BiLSTM_layer(hidden_dim_list=hidden_dim)
        else:
            self.BiLSTM_layer(hidden_dim[0])
        self.logits = self.attention(self.logits)
        self.CRF_layer()
        
        #优化器
        optim = tf.train.AdamOptimizer(learning_rate=self.lr)
        grads_and_vars = optim.compute_gradients(self.loss)
        grads_and_vars_clip = [[tf.clip_by_value(g, -clip_grad, clip_grad), v] for g, v in grads_and_vars]
        self.train_op = optim.apply_gradients(grads_and_vars_clip)
#         self.train_op = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)
        
        #初始化图中的变量
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = 0.4  # need ~700MB GPU memory
        self.sess = tf.Session(config=config)
        self.sess.run(tf.global_variables_initializer())
                      
        #CRF层
    def CRF_layer(self):
        log_likelihood, self.transition_params = \
                   crf_log_likelihood(inputs=self.logits,tag_indices=self.real_labels, sequence_lengths=self.seq_length)
#         class_weight_data = tf.Variable(run_time.label_weight_array, dtype=tf.float32, trainable=False)
#         self.ts_class_weight_ = tf.nn.embedding_lookup(params=class_weight_data, ids=self.real_labels)
#         self.ts_class_weight = tf.reduce_mean(self.ts_class_weight_, axis=1)
#         self.flag = tf.multiply(log_likelihood, self.ts_class_weight)
#         loss = -tf.reduce_mean(tf.multiply(log_likelihood, self.ts_class_weight))
        self.loss = -tf.reduce_mean(log_likelihood)
        
    def BiLSTM_layer(self, hidden_dim):
 
        with tf.variable_scope("bilstm_" + str(0)):
            cell_fw, cell_bw = LSTMCell(hidden_dim), LSTMCell(hidden_dim)
            (output_fw_seq, output_bw_seq), _ = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=cell_fw, cell_bw=cell_bw, inputs=self.word_embeddings, sequence_length=self.seq_length, dtype=tf.float32)
            output = tf.concat([output_fw_seq, output_bw_seq], axis=-1)
            output = tf.nn.dropout(output, self.dropout)
#         is_training = True
#         output = tf.layers.batch_normalization(output, training=is_training)
        with tf.variable_scope("proj_" + str(0)):
            self.W = tf.get_variable(name="W_" + str(0),shape=[2 * hidden_dim, run_time.TAG_NUM],
                                initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
            self.b = tf.get_variable(name="b_" + str(0), shape=[run_time.TAG_NUM],
                                initializer=tf.zeros_initializer(), dtype=tf.float32)
        s = tf.shape(output)
        output = tf.reshape(output, [-1, 2*hidden_dim])
        pred = tf.matmul(output, self.W) + self.b
        self.logits = tf.reshape(pred, [-1, s[1], run_time.TAG_NUM])
    
    def multi_BiLSTM_layer(self, hidden_dim_list = [50, 50, 50]): 
        print("lstm结构是", hidden_dim_list)
        inversed_inputs = tf.reverse_sequence(self.word_embeddings, self.seq_length, batch_dim=0, seq_axis=1)
        def attn_cell(n_hidden):
            lstm_cell = LSTMCell(n_hidden, forget_bias=0.8)#tf.contrib.rnn.BasicLSTMCell(n_hidden)
            return tf.contrib.rnn.DropoutWrapper(lstm_cell, output_keep_prob=self.dropout)

        cell_fw_list, cell_bw_list = [], []
        for hidden_dim in hidden_dim_list: 
            cell_fw_list.append(attn_cell(hidden_dim))
            cell_bw_list.append(attn_cell(hidden_dim)) 
#         mlstm_cell_fw = tf.contrib.rnn.MultiRNNCell(cell_fw_list, state_is_tuple=True)
#         mlstm_cell_bw = tf.contrib.rnn.MultiRNNCell(cell_bw_list, state_is_tuple=True)
        print("lstm的层数是", len(cell_bw_list))
        mlstm_cell_fw = tf.nn.rnn_cell.MultiRNNCell(cell_fw_list, state_is_tuple=True)
        mlstm_cell_bw = tf.nn.rnn_cell.MultiRNNCell(cell_bw_list, state_is_tuple=True)
        initial_state_fw = mlstm_cell_fw.zero_state(self.batch_size, dtype=tf.float32)
        initial_state_bw = mlstm_cell_bw.zero_state(self.batch_size, dtype=tf.float32)
        
        with tf.variable_scope("bilstm_fw"):#正向
            output_fw_0, _ = tf.nn.dynamic_rnn(cell=mlstm_cell_fw, inputs=self.word_embeddings, initial_state=initial_state_fw,
                sequence_length=self.seq_length, dtype=tf.float32)
            
        with tf.variable_scope("bilstm_bw"):#反向
            output_bw_0, _ = tf.nn.dynamic_rnn(cell=mlstm_cell_bw, inputs=inversed_inputs, initial_state=initial_state_bw, 
                sequence_length=self.seq_length, dtype=tf.float32)
        
        output = tf.concat([output_fw_0, output_bw_0], axis=-1)
#         output = tf.nn.dropout(output, drop_out)
        layer_no = 0
        s = tf.shape(output)
        with tf.variable_scope("proj_" + str(layer_no)):
            self.W = tf.get_variable(name="W_" + str(layer_no),shape=[2 * hidden_dim, run_time.TAG_NUM],
                                initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
 
            self.b = tf.get_variable(name="b_" + str(layer_no), shape=[run_time.TAG_NUM],
                                initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
         
        output = tf.reshape(output, [-1, 2*hidden_dim])
        pred = tf.matmul(output, self.W) + self.b
#         pred = tf.nn.softmax(pred)#增加softmax，扩大输出之间的差距，是否有收益呢？
        self.logits = tf.reshape(pred, [-1, s[1], run_time.TAG_NUM])

    #https://blog.csdn.net/huanghaocs/article/details/85255227
    def attention(self, inputs, attention_size=100, time_major=False):
        if isinstance(inputs, tuple):
            inputs = tf.concat(inputs, 2)
        if time_major:  # (T,B,D) => (B,T,D)
            inputs = tf.transpose(inputs, [1, 0, 2])
        hidden_size = inputs.shape[2].value 
        # Trainable parameters
        w_omega = tf.Variable(tf.random_normal([hidden_size, attention_size], stddev=0.1))
        b_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1))
        u_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1))
        v = tf.tanh(tf.tensordot(inputs, w_omega, axes=1) + b_omega)
    
        vu = tf.tensordot(v, u_omega, axes=1, name='vu')  # (B,T) shape
        alphas = tf.nn.softmax(vu, name='alphas')  # (B,T) shape

        #the result has (B, T, D) shape
        output = tf.multiply(inputs, tf.expand_dims(alphas, -1))
        output = tf.add(output,inputs)#改动点
#         output = tf.transpose(output, perm=[1, 0, 2])
        return output
            
    def lr_decay(self, init_lr, step): return init_lr*np.exp(-0.02*step**0.6)
    
    def fit(self, epoch_num=100, init_lr=0.01, batch_size=50, app_name="train", lr_decay_strategy="const"):
        self.mode = 'train'
        saver = tf.train.Saver(tf.global_variables())
        global_step = 0
        data_list = data_loader.read_corpus(run_time.PATH_TRAINING_CORPUS, )
        test_data_list = data_loader.read_corpus(run_time.PATH_TESTING_CORPUS)
        testing_data = data_loader.batch_yield(test_data_list,batch_size, self.word_id_map, run_time.TAG_LABEL_MAP, shuffle=True)
            
        for epoch in range(epoch_num):
            random.shuffle(data_list)
            traing_data = data_loader.batch_yield(data_list, batch_size, self.word_id_map, run_time.TAG_LABEL_MAP, shuffle=True)
            for step, (seqs, labels) in enumerate(traing_data):
#                 for line in labels: print(line)
                global_step += 1
                if lr_decay_strategy=="const":
                    current_lr = init_lr
                elif lr_decay_strategy=='decay':
                    current_lr = self.lr_decay(init_lr, global_step)
                else:
                    if epoch<10:
                        current_lr = 0.01
                    else:
                        current_lr = 0.001
                data_dict, _ = self.get_feed_dict(seqs, labels=labels, lr=current_lr, dropout=0.5)
                _, loss_train = self.sess.run([self.train_op, self.loss], feed_dict=data_dict)
                if step%20==0:
                    print(epoch, step,'/', len(traing_data) , "loss_train", loss_train, current_lr)
            if epoch%5==0:
                print("训练集:")
                self.evaluate(traing_data[:10], app_name)
                print("测试集:")
                self.evaluate(testing_data, app_name)

    def get_feed_dict(self, seqs, labels=None, lr=None, dropout=None):
        """
        seqs:词语id序列，labels:标签id, lr:学习率
        :return: feed_dict
        """
        word_ids, seq_len_list = pad_sequences(seqs, pad_mark=0)
        feed_dict = {self.word_ids: word_ids, self.seq_length: seq_len_list, self.batch_size: len(seqs)}
        if labels is not None:
            labels_, _ = pad_sequences(labels, pad_mark=0)
            feed_dict[self.real_labels] = labels_
        if lr is not None: feed_dict[self.lr] = lr
        if dropout is not None: feed_dict[self.dropout] = dropout
        return feed_dict, seq_len_list

    def predict(self):
        pass
       
    def evaluate(self, testing_data, app_name):
        all_pred_tags, all_real_tags, all_chars = [], [], []
        print("开始测试")
        for step, (seq_list, labels_list) in enumerate(testing_data):
            data_dict, seq_len_list = self.get_feed_dict(seq_list, labels=labels_list, dropout=1.0)
            logits, transition_params = self.sess.run([self.logits, self.transition_params],
                                             feed_dict=data_dict)
#             print('transition_params', transition_params)
            if step%200==0: print("测试进度是", step, '/', len(testing_data))
            for logit, seq_len, labels, seq in zip(logits, seq_len_list, labels_list, seq_list):
                viterbi_seq, _ = viterbi_decode(logit[:seq_len], transition_params)
                pred_tag = [run_time.LABEL_TAG_MAP[label] for label in viterbi_seq]
                real_tags = [run_time.LABEL_TAG_MAP[label] for label in labels]
                chars = [self.id_word_map[label] for label in seq]
                all_pred_tags += pred_tag
                all_real_tags += real_tags
                all_chars += chars
        all_pred_tags = list(map(lambda x: '0' if x=="O" else x, all_pred_tags))
        all_real_tags = list(map(lambda x: '0' if x=="O" else x, all_real_tags))
        lines = [all_chars[i] + ' ' + all_real_tags[i] + ' ' + all_pred_tags[i] + '\n' \
                               for i in range(len(all_pred_tags))]
        test_result_path = '../../data/output/' + app_name + '_test_label.txt'
        test_metric_path = '../../data/output/' + app_name + '_metric.txt'
        print("开始计算指标取值")
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

