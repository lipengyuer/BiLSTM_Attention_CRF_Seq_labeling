# 加载数据或者模型
import sys, os, time

path = os.getcwd()
path = os.path.dirname(path)
sys.path.append(path)
import pymysql

from database.get_connection import get_connection_mysql
from config import environment_configure
import requests, json


# 加载地域名称数据
def load_region_code():
    conn = get_connection_mysql(db='common_tag')
    cur = conn.cursor()
    sql_str = "select region_id, region_name, all_name, level, continent, continent_code, country, country_code, province,  province_code, city, city_code from dic_region"
    cur.execute(sql_str)
    data = cur.fetchall()

    region_id_map = {}
    for line in data:
        region_id, region_name, all_name, level, continent, continent_code, \
        country, country_code, province, province_code, city, city_code = line

        data = {'regionId': region_id, 'regionName': region_name, 'level': level, \
                'continent': continent, 'continentCode': continent_code, \
                'country': country, 'countryCode': country_code, 'province': province, \
                'provinceCode': province_code, 'city': city, 'cityCode': city_code}
        if continent == None:  # 大洲字段为空，是中国地名
            data['continent'] = '亚洲'
            data['continentCode'] = 1
        for key in list(data.keys()):
            if data[key] == None:
                del data[key]

        region_id_map[all_name] = region_id_map.get(region_name, []) + [data]
    # print(region_id_map)
    cur.close()
    conn.close()
    return region_id_map


# 加载人名数据
def load_person_name_code():
    conn = get_connection_mysql(db=environment_configure.MYSQL_DB_DAOGU)
    cur = conn.cursor()
    sql_str = "select id, name from refining_entity_dictionary where entity='rm'"
    cur.execute(sql_str)
    data = cur.fetchall()

    person_name_id_map = {}
    for line in data:
        person_name_id_map[line[1]] = person_name_id_map.get(line[1], []) + [line[0]]
    cur.close()
    conn.close()
    return person_name_id_map


# 加载主题词数据
def load_topic_word_code():
    conn = get_connection_mysql(db=environment_configure.MYSQL_DB_DAOGU)
    cur = conn.cursor()
    sql_str = "select id, parent_id, name from refining_entity_dictionary where entity='szdy'"
    cur.execute(sql_str)
    data = cur.fetchall()

    topic_word_id_map = {}
    for line in data:
        topic_word_id_map[line[2]] = topic_word_id_map.get(line[1], []) + [{'id': line[0], 'parent_id': line[1]}]
    cur.close()
    conn.close()
    return topic_word_id_map


# 加载文章的缩略图，启动的时候加载到内存中，然后查询，相当于是写死的。如果有新文章进来，需要改为直接从mysql中查询。
def load_simple_image_of_article_code():
    conn = get_connection_mysql(db='trswcm')
    cur = conn.cursor()
    sql_str = "SELECT APPDOCID,APPFILE FROM wcmappendix"
    cur.execute(sql_str)
    data = cur.fetchall()

    APPFILE_articleId_map = {}
    for line in data:
        APPFILE_articleId_map[line[0]] = APPFILE_articleId_map.get(line[0], []) + [line[1]]
    cur.close()
    conn.close()
    return APPFILE_articleId_map


def load_entity_names():
    url = 'http://172.18.89.13:9010/ontokb/getAllNamedIndividuals'
    resp = requests.get(url)
    entity_set = set(json.loads(resp.text))
    url = 'http://172.18.89.13:9010/ontokb/getAllDataProperties'
    resp = requests.get(url)
    property_set = set(json.loads(resp.text))
    url = 'http://172.18.89.13:9010/ontokb/getAllObjectProperties'
    resp = requests.get(url)
    relation_set = set(json.loads(resp.text))
    return entity_set, property_set, relation_set


# 加载习30讲字符串及其对应编码，用于文本分类中习思想维度
def load_xi_30_speeches_code():
    conn = get_connection_mysql(db='common_tag')
    cur = conn.cursor()
    sql_str = "select label_id, parent_id, name from dic_label where parent_id=18"
    cur.execute(sql_str)
    data = cur.fetchall()

    xi_30_speeches_id_map = {}
    for line in data:
        xi_30_speeches_id_map[line[2]] = {'id': line[0], 'parent_id': line[1]}
    cur.close()
    conn.close()
    return xi_30_speeches_id_map


# 加载组织名称及其简称数据
def load_orgnization_names():
    conn = get_connection_mysql(host='172.18.89.14', user='root', passwd='Founder123', db=environment_configure.MYSQL_DB_DAOGU)
    cur = conn.cursor()
    sql_str = "select id,full_entity,abb_entity from abb_entity_dictionary where category='jg'"
    cur.execute(sql_str)
    name_id_map = {}
    shorter_full_name_map = {}
    for row in cur:
        shorter_name = row[2].split(', ')
        full_name = row[1]
        id = row[0]
        name_id_map[full_name] = id
        for a_name in shorter_name:
            shorter_full_name_map[a_name] = shorter_full_name_map.get(a_name, []) + [full_name]
    cur.close()
    conn.close()
    return name_id_map, shorter_full_name_map


# 加载会议名称-id及其简称-全称数据
def load_conference_names():
    conn = get_connection_mysql(host='172.18.89.14', user='root', passwd='Founder123', db=environment_configure.MYSQL_DB_DAOGU)
    cur = conn.cursor()
    sql_str = "select id,full_entity,abb_entity from abb_entity_dictionary where category='hy'"
    cur.execute(sql_str)
    data = cur.fetchall()
    name_id_map = {}
    shorter_full_name_map = {}
    for line in data:
        shorter_name = line[2].split(', ')
        full_name = line[1]
        id = line[0]
        name_id_map[full_name] = id
        for a_name in shorter_name:
            shorter_full_name_map[a_name] = shorter_full_name_map.get(a_name, []) + [full_name]
    cur.close()
    conn.close()
    return name_id_map, shorter_full_name_map


# 加载文章id和活动的分类及其id
def load_doc_id_hd_classification():
    # 读取每个活动类型的id
    conn = get_connection_mysql(db='common_tag')
    cur = conn.cursor()
    sql_str = "select label_id, parent_id, name from dic_label where parent_id=15"
    cur.execute(sql_str)
    data = cur.fetchall()
    hd_cat_id_map = {}
    for line in data:
        hd_cat_id_map[line[2]] = line[0]
    cur.close()
    conn.close()

    conn = get_connection_mysql(db=environment_configure.MYSQL_DB_DAOGU)
    cur = conn.cursor()
    sql_str = "select doc_id, hd_category from resource_document"
    cur.execute(sql_str)
    data = cur.fetchall()
    docId_cat_map = {}
    for line in data:

        if line[1] != None:
            docId_cat_map[line[0]] = {'name': line[1], 'id': hd_cat_id_map[line[1]]}
    cur.close()
    conn.close()

    return docId_cat_map

def load_region_name_data_for_normalization():
    conn = get_connection_mysql(db='common_tag')
    cur = conn.cursor()
    sql_str = "select region_id, region_name, region_short, level, continent, continent_code, country_short, country_code, province,  province_code, city, city_code from dic_region where level<=4"
    cur.execute(sql_str)
    data = cur.fetchall()

    region_id_map = {}
    region_short_map = {}
    for line in data:
        #提取原始数据
        region_id, region_name, region_short, level, continent, continent_code, \
        country_short, country_code, province, province_code, city, city_code = line

        data = {'regionId': region_id, 'regionName': region_name, 'level': level, \
                'continent': continent, 'continentCode': continent_code, \
                'country': country_short, 'countryCode': country_code, 'province': province, \
                'provinceCode': province_code, 'city': city, 'cityCode': city_code}
        if continent == None:  # 大洲字段为空，是中国地名
            data['continent'] = '亚洲'
            data['continentCode'] = 1
        for key in list(data.keys()):
            if data[key] == None:
                del data[key]
        
        #记录标准名称-详细地名
        region_id_map[region_name] = region_id_map.get(region_name, []) + [data]
        #记录别称-标准名称
        if region_short!=None:
            alter_names = region_short.split(',')
            for name in alter_names:
                region_short_map[name] = region_short_map.get(name , []) + [region_name]

    cur.close()
    conn.close()
#     print(region_short_map)
    return region_id_map, region_short_map   

def load_all_region_names():
    conn = get_connection_mysql(db='common_tag')
    cur = conn.cursor()
    sql_str = "select region_name from dic_region where level<=4"
    cur.execute(sql_str)
    data = cur.fetchall()
    region_name_list = []
    for line in data:
        #提取原始数据
        region_name = line[0]
        region_name_list.append(region_name)
    cur.close()
    conn.close()
    return region_name_list

def load_all_entities():
    words = load_all_region_names()
    words = set(words)
    conn = get_connection_mysql(db=environment_configure.MYSQL_DB_DAOGU)
    cur = conn.cursor()
    sql_str = "select abb_entity from abb_entity_dictionary"
    cur.execute(sql_str)
    data = cur.fetchall()
    for line in data:
        shorter_name = line[0].split(', ')
        for a_name in shorter_name:
            words.add(a_name)
            
    sql_str = "select name from refining_entity_dictionary where status='ok'"
    cur.execute(sql_str)
    data = cur.fetchall()
    for line in data:
            words.add(line[0])               
    cur.close()
    conn.close()
    return words

def load_all_region_short():
    conn = get_connection_mysql(db='common_tag')
    cur = conn.cursor()
    sql_str = "select region_short from dic_region where country='中华人民共和国'"
    cur.execute(sql_str)
    data = cur.fetchall()
    region_name_list = []
    for line in data:
        #提取原始数据
        region_short = line[0]
        if region_short==None : continue
        region_shorts = region_short.split(",")
        for name in region_shorts:
            region_name_list.append(name)
    cur.close()
    conn.close()
    return region_name_list

if __name__ == '__main__':
    # load_orgnization_names()
    #load_doc_id_hd_classification()
    #load_region_name_data_for_normalization()
    _, result = load_region_name_data_for_normalization()
    print(result["Jamaica"])
