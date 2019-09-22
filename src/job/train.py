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
    seq_lab_model = model.BiLSTMCRF(use_pretrained_embeddings=False, embedding_dim=150, hidden_dim = 200)
    seq_lab_model.fit(epoch_num=50, init_lr=0.001)
    


if __name__ == '__main__':
    train_model()