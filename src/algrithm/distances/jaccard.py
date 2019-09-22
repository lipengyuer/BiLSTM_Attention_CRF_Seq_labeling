'''
Created on Apr 12, 2017

@author: foxbat
'''
def calcJaccard(set1,set2):
    return 1-len(set1&set2)/len(set1|set2)
