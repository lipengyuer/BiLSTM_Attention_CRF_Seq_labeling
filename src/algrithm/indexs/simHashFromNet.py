'''
Created on May 26, 2017

@author: foxbat
'''

class simhash:  
     
    #构�?�函�?  
    def __init__(self, tokens='', hashbits=128):         
        self.hashbits = hashbits  
        self.hash = self.simhash(tokens);  
     
    #toString函数     
    def __str__(self):  
        return str(self.hash)  
     
    #生成simhash�?     
    def simhash(self, tokens):  
        v = [0] * self.hashbits  
        for t in [self._string_hash(x) for x in tokens]: #t为token的普通hash�?            
            for i in range(self.hashbits):
                bitmask = 1 << i
                if t & bitmask :
                    v[i] += 1 #查看当前bit位是否为1,是的话将该位+1  
                else:
                    v[i] -= 1 #否则的话,该位-1  
        fingerprint = 0
        for i in range(self.hashbits):
            if v[i] >= 0:
                fingerprint += 1 << i
        return fingerprint #整个文档的fingerprint为最终各个位>=0的和  
     
    #求海明距�?  
    def hamming_distance(self, other):  
        x = (self.hash ^ other.hash) & ((1 << self.hashbits) - 1)  
        tot = 0;  
        while x :  
            tot += 1  
            x &= x - 1
        return tot
     
    #求相似度  
    def similarity (self, other):  
        a = float(self.hash)  
        b = float(other.hash)  
        if a > b : return b / a
        else: return a / b

     
    #针对source生成hash�?   (�?个可变长度版本的Python的内置散�?)  
    def _string_hash(self, source):         
        if source == "":  
            return 0  
        else:  
            x = ord(source[0]) << 7  
            m = 1000003  
            mask = 2 ** self.hashbits - 1  
            for c in source:  
                x = ((x * m) ^ ord(c)) & mask  
            x ^= len(source)
            if x == -1:  
                x = -2  
            return x  
              
if __name__ == '__main__':  
    s = '把�??�?要�??判断�?文本�?分词�?形成�?这�??个�??文章�?的�??特征�?单词�?。�??�?后�??形成�?去�??掉�??噪音�?词�??的�??单词�?序列�?并�??为�??每个�?词�??加上�?权重'
    
    hash1 = simhash(s.split())  
     
    s = '把�??�?要�??判断�?文本�?分词�?形成�?这�??个�??文章�?的�??特征�?单词�?。�??�?后�??形成�?去�??掉�??为�??每个�?词�??加上�?权重'  
    hash2 = simhash(s.split())  
     
    s = '中国 人民 , 水电费�??非�??婆婆�?连连看�??据ｕ�?文石'
    hash3 = simhash(s.split())  
     
    print(hash1.hamming_distance(hash2) , "   " , hash1.similarity(hash2))  
    print(hash1.hamming_distance(hash3) , "   " , hash1.similarity(hash3)) 
    
    print(hash1.hamming_distance(hash2)/128 , "   " , hash1.similarity(hash2))  
    print(hash1.hamming_distance(hash3)/128 , "   " , hash1.similarity(hash3))