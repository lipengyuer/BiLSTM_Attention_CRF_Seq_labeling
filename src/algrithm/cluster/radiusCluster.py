'''
Created on Sep 22, 2017

@author: foxbat
'''
import copy

from algrithm.distances import angleCosin
from algrithm.indexs import invertedIndex


# 微聚类算法：
# 输入dataMap为{'2345':[['中国'�?34],['日本',12]],......}
# 输出为一组clusterList
# 其中�?个cluster为：id
#                  center {'id':centerDataId,'sparse':centerDataSparse}
#                  oldCenter {'id':centerDataId,'sparse':centerDataSparse}
#                  oldCenterSparse
class microCluster():

    def __init__(self):
        pass
        # self.id                                 类ID
        # self.center                             类中心的文章对象
        # self.oldCenter�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?原来类中心的文章对象
        # self.members = members�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?[(类成员id,距离)]
        # self.weight


def delEmptyCluster(clustersMap):
    # 删除在聚类过程中产生的空�?
    emptyClusters = []
    for clusterId in clustersMap:
        cluster = clustersMap[clusterId]
        if len(cluster.members) == 0:
            emptyClusters.append(clusterId)
    for clusterId in emptyClusters:
        del clustersMap[clusterId]


def insertASparse(article, radius, clusterMap, iindex, clusterIdSet):
    # 插入�?条记�?
    searchList = invertedIndex.getSearchSet(article.sparse, iindex)
    insertToNew = 1
    insertToCluster = 0
    # 如果找到合�?�的簇，加入
    for clusterId in searchList:
        dist = angleCosin.calSparseCosinDistance(article.sparseMap, article.length, clusterMap[
                                                 clusterId].center.sparseMap, clusterMap[clusterId].center.length)
        if dist <= radius:
            insertToCluster = clusterId
            insertToNew = 0
            break
    if insertToNew == 0:
        # print(article.id)
        # print(clusterMap[insertToCluster].members)
        clusterMap[insertToCluster].members[article.id] = dist
        #clusterMap[insertToCluster].weight += article.weight
    # 如果找不到，就新建一个簇，并以本文章作为簇中�?
    else:
        cluster = microCluster()
        cluster.members = {article.id: 0.0}
        newId = publicService.getNoDupInt(clusterIdSet, 100000000)
        clusterIdSet.add(newId)
        cluster.id = newId
        cluster.center = article
        clusterMap[newId] = cluster
        insertToCluster = newId

        articleTemp = ArticleDto()
        articleTemp.id = newId
        articleTemp.sparse = article.sparse
        invertedIndex.insert(articleTemp.id, articleTemp.sparse, iindex)

    return insertToNew, insertToCluster


def buildIIndex(clustersMap):
    iiList = []
    for clusterId in clustersMap:
        articleTemp = ArticleDto()
        articleTemp.id = clusterId
        articleTemp.sparse = clustersMap[clusterId].center.sparse
        iiList.append(articleTemp)
    iindex = invertedIndex.buildInvertedIndex(iiList)
    return iindex


def insertSparse(dataMap, radius, clusterMap):
    # 向clusters 中加入数�?
    for clusterId in clusterMap:
        cluster = clusterMap[clusterId]
        cluster.members = {}
    iindex = buildIIndex(clusterMap)
    clusterIdSet = set(clusterMap.keys())
    ids = list(dataMap.keys())
    for articleId in ids:
        insertASparse(
            articleMap[articleId], radius, clusterMap, iindex, clusterIdSet)


def updateSparseCenter(articlesMap, clustersMap):
    # 更新类中�?
    for articleId in clustersMap:
        cluster = clustersMap[articleId]
        cluster.oldCenter = cluster.center
    for articleId in clustersMap:
        cluster = clustersMap[articleId]
        if len(cluster.members) > 1:
            memberList = []
            for memberId in cluster.members:
                memberList.append(articlesMap[memberId])
            newCenter = angleCosin.getCenterArticle(memberList)
            if newCenter.id != cluster.center.id:
                cluster.center = copy.deepcopy(newCenter)


def calMaxSparseOffset(articlesMap, clustersMap):
    # 计算类中心偏移量
    centerShift = 0
    for articleId in clustersMap:
        cluster = clustersMap[articleId]
        if len(cluster.members) > 1:
            center = cluster.center
            oldCenter = cluster.oldCenter
            shift = angleCosin.calSparseCosinDistance(
                center.sparseMap, center.length, oldCenter.sparseMap, oldCenter.length)
            if shift > centerShift:
                centerShift = shift
    return centerShift


def buildSparseCluster(dataMap, radius, rounds, offset):
    # dataMap:�?有有关文章的字典｛id : sparse�?
    # radius:聚类的最大半�?
    # round:�?多允许迭代多少轮�?
    # offset:能够容忍的最大的聚类中心偏移�?
    clusterMap = {}
    loopCount = 0
    maxOffset = 1
    conditions = True

    while conditions:
        insertSparse(dataMap, radius, clusterMap)
        delEmptyCluster(clusterMap)
        updateSparseCenter(dataMap, clusterMap)
        loopCount += 1
        maxOffset = calMaxSparseOffset(dataMap, clusterMap)
        if loopCount >= rounds or maxOffset < offset:
            conditions = False
    return clusterMap
