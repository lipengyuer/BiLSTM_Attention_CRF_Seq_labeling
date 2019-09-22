'''
Created on 2019年8月10日

@author: Administrator
'''
from  pymongo import MongoClient

def getTextMining():
    conn = MongoClient('192.168.1.106', 27017)
    db = conn["textMining"]
    #db.authenticate("foxbat", "foxbat")
    return db

if __name__ == '__main__':
    conn = getTextMining()
    lines =conn['FamilyName_300'].find()
    lines = list(map(lambda x: x['name'] + '\n', lines))
    print(lines)
    with open('FamilyName_300.txt', 'w', encoding='utf8') as f:
        f.writelines(lines)
    