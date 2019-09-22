'''
Created on Jul 22, 2016

@author: foxbat
'''
# 倒查索引
# 输入数据格式为[{'id':2354,'sparse':[['中国'�?34],['日本',12]]}]
# iindex格式为：{'中国'：[2345,2205],'日本':[2345]}


def insert(dataId, sparse, iindex):
    for tup in sparse:
        if tup[0] in iindex:
            iindex[tup[0]].add(dataId)
        else:
            idSet = set()
            idSet.add(dataId)
            iindex[tup[0]] = idSet


def delete(dataId, sparse, iindex):
    for tup in sparse:
        if tup[0] in iindex:
            if dataId in iindex[tup[0]]:
                iindex[tup[0]].remove(dataId)
            if len(iindex[tup[0]]) == 0:
                del iindex[tup[0]]


def buildInvertedIndex(dataList):
    iindex = {}
    for data in dataList:
        insert(data['id'], data['sparse'], iindex)
    return iindex


def getSearchSet(sparse, iindex):
    searchSet = set()
    for tup in sparse:
        if tup[0] in iindex:
            searchSet = searchSet | iindex[tup[0]]
    return searchSet

def getSameSim(sparse, iindex):
    low_same = len(sparse)*0.8
    searchMap = {}
    for tup in sparse:#遍历当前新闻的关键词
        if tup[0] in iindex:#如果这个关键词在iindex的key集合�?
            for tempId in iindex[tup[0]]:#遍历包含这个关键词的�?有新闻id
                if tempId in searchMap:
                    searchMap[tempId] += 1#统计包含当前新闻,�?包含的关键词的个�?
                else:
                    searchMap[tempId] = 1
    sameList = []
    for tempId in searchMap:
        if searchMap[tempId] >= low_same:
            sameList.append(tempId)
    return sameList

