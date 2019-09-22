'''
Created on Jul 22, 2016

@author: foxbat
'''
import math

import numpy as np


def calcDenseLength(dense):
    #计算密集向量的长�?
    length = math.sqrt(np.dot(dense,dense))
    return length

def calDenseCosinDistance(dense1,length1,dense2,length2):
    #计算两个密集向量的余弦夹角距�?
    cosin = 0
    if length1==0 and length2==0:
        cosin = 1
    elif length1==0 or length2==0:
        cosin = 0
    else:
        xy = np.dot(dense1,dense2)
        cosin = xy/(length1*length2)
    return 1-cosin;

def sparse2Map(sparse):
    #生成�?疏向量Map,便于计算两个�?疏向量交�?
    sparseMap = {}
    for tup in sparse:
        sparseMap[tup[0]] = tup[1]
    return sparseMap

def calcSparseLength(sparse):
    #计算�?疏向量的长度
    x = [tup[1] for tup in sparse]
    length = math.sqrt(np.dot(x,x))
    return length


def calSparseCosinDistance(sparseMap1,length1,sparseMap2,length2):
    cosin = 0
    if length1==0 and length2==0:
        cosin = 1
    elif length1==0 or length2==0:
        cosin = 0
    else:
        set1 = sparseMap1.keys()
        set2 = sparseMap2.keys()
        crossList = list(set1&set2)
        if len(crossList)==0:
            cosin = 0
        else:
            xCross = [sparseMap1[dim] for dim in crossList]
            yCross = [sparseMap2[dim] for dim in crossList]
            xy = np.dot(xCross,yCross)
            cosin = xy/(length1*length2)
    return 1-cosin


def getCenterArticle(articleList):
    #�?组稀疏向量的中心�?
    centerMap = {}
    for article in articleList:
        for tup in article.sparse:
            if tup[0] in centerMap:
                centerMap[tup[0]] += tup[1]*article.weight
            else:
                centerMap[tup[0]] = tup[1]*article.weight
                
    count = len(articleList)
    for dim in centerMap:
        centerMap[dim] /= count
    centerLength = 0
    for dim in centerMap:
        centerLength += centerMap[dim]**2
    centerLength = math.sqrt(centerLength)
    
    minDist = 1
    center = 0
    for article in articleList:
        dist = calSparseCosinDistance(article.sparseMap,article.length,centerMap,centerLength)
        if dist < minDist:
            minDist = dist
            center = article
    return center



def getDenseCenter(denses):
    #�?组密集向量的中心�?
    dCenter = denses.average(axis = 0)
    return dCenter

'''def getDenseCenterId(articles):
    #寻找中心�?
    denses = np.asarray([article.dense for article in articles])
    center = getDenseCenter(denses)
    centerId = 0
    distance = 1
    for article in articles:
        tempDist = calCosin(article.dense,center)
        if tempDist < distance:
            distance = tempDist
            centerId = article.id
    return centerId'''

