'''
Created on 2019年8月10日

@author: Administrator
'''
from  pymongo import MongoClient
from config import environment_configure
import pymysql

def getTextMining():
    conn = MongoClient('172.18.89.13', 27017)
    db = conn["textMining"]
    #db.authenticate("foxbat", "foxbat")
    return db

def get_connection_mysql(host=environment_configure.MYSQL_IP,
                           port=environment_configure.MYSQL_PORT, 
                           user=environment_configure.MYSQL_USER,
                           passwd=environment_configure.MYSQL_PASSWORD,
                           charset=environment_configure.MYSQL_CHATSET,
                           db=environment_configure.MYSQL_DB):
    conn = pymysql.connect(host=host, port=port,  user=user,
                           passwd=passwd, charset=charset, db=db)
    return conn