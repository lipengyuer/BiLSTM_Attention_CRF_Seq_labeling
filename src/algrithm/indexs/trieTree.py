'''
Created on Nov 3, 2016

@author: foxbat
'''
def buildTrieTree(dictionary):
    trieTree = {}
    trieTree['isEnd'] = 0
    pointer = trieTree
    for word in dictionary:
        for char in word:
            if char not in pointer:
                pointer[char] = {}
                pointer[char]['isEnd'] = 0
            pointer = pointer[char]
        pointer['isEnd'] = 1
        pointer = trieTree
    return trieTree


def search(trieTree,content):
    positions = []
    words = []
    contentLength = len(content)
    for i in range(contentLength):
        if content[i] in trieTree:
            pointer = trieTree
            offSet = 0
            word = ''
            tempWord = ''
            while i+offSet < contentLength:
                char = content[i+offSet]
                if char in pointer:
                    tempWord += char
                    if pointer[char]['isEnd'] == 1:
                        word += tempWord
                        tempWord = ''
                    pointer = pointer[char]
                else:
                    break
                offSet += 1
            if len(word) > 0:
                positions.append(i)
                words.append(word)
    return positions,words
            