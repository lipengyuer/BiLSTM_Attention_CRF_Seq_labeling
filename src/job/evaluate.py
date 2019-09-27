'''
Created on 2019年9月21日

@author: Administrator
'''
import os
import sys


path = os.getcwd()
path = os.path.dirname(path)
sys.path.append(path)
from job.data_preprocess import load_corpus
def check_error_in_corpus():
    chars_tags_file_list = load_corpus()
    for chars, tags, file_name in chars_tags_file_list:
        text = ''.join(chars)
        if "德国有史以来第一位" in text:#
            print(file_name)
    
    
if __name__ == '__main__':
    check_error_in_corpus()
    