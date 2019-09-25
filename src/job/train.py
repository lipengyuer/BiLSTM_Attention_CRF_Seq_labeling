'''
Created on 2019年9月21日

@author: Administrator
'''
import os
import sys
path = os.getcwd()
path = os.path.dirname(path)
sys.path.append(path)
from model import model
from config import run_time

def train_model():
    print("加载训练数据")
    app_name = sys.argv[1]
    gpu_no = sys.argv[2]
    lr_decay_strategy = sys.argv[3]
    if_multi_layer = sys.argv[4]
        ## Session configuration
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_no)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # default: 0
    init_lr = '0.001'
    seq_lab_model = model.BiLSTMCRF(use_pretrained_embeddings=True, embedding_dim=100, hidden_dim = [150, 100], if_multi_layer=int(if_multi_layer))
    seq_lab_model.fit(epoch_num=1000, init_lr=float(init_lr), batch_size=200, app_name=app_name, lr_decay_strategy=lr_decay_strategy)
    
if __name__ == '__main__':
    train_model()