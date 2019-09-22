'''
Created on May 26, 2017

@author: foxbat
'''
#print(bin(ord('�?')))

def string_hash(source,hashBits):
    if source == "":
        return 0
    else:
        x = ord(source[0]) << 7
        #print(bin(ord(source[0])))
        #print(bin(x))
        m = 1000003
        mask = 2 ** hashBits - 1
        #print(bin(mask))
        for c in source:
            x = ((x * m) ^ ord(c)) & mask
            #print(bin(x*m))
            #print(bin(ord(c)))
            #print(bin((x * m) ^ ord(c)))
            #print(bin(x))
        x ^= len(source)
        if x == -1:
            x = -2
        return x

print(bin(string_hash('北京',64)))