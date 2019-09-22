'''
Created on May 26, 2017

@author: foxbat
'''
import hashlib

from entity.article import ArticleDto


def wordHash(word):
    word = word.encode("utf8")
    h=hashlib.md5()
    h.update(word)
    hashCode=h.hexdigest()
    return hashCode
#bin(int(str,16))
print(bin(int(wordHash('北京'),16)))

def getKeyWords(content):
    article = ArticleDto()
    article.id = 1
    article.title = ''
    article.content = content
    
    article.filtNoise()
    article.article2Sentence()
    article.sentenceSplitWords()
    article.words()
    article.sentenceWordWeights()
    article.wordWeightMap()
    article.keyWords()
    
    return article.keyWords

def getSimHashCode(content):
    keyWords = getKeyWords(content)
    for tup in keyWords:
        pass
        
    
    
    
    
    
    