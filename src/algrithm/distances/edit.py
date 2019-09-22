'''
Created on Feb 27, 2017

@author: foxbat
'''
def levenshtein(first,second):  
    if len(first) > len(second):  
        first,second = second,first  
    if len(first) == 0:  
        return len(second)  
    if len(second) == 0:  
        return len(first)  
    first_length = len(first) + 1  
    second_length = len(second) + 1  
    distance_matrix = [list(range(second_length)) for x in range(first_length)]   
    for i in range(1,first_length):  
        for j in range(1,second_length):  
            deletion = distance_matrix[i-1][j] + 1  
            insertion = distance_matrix[i][j-1] + 1  
            substitution = distance_matrix[i-1][j-1]  
            if first[i-1] != second[j-1]:  
                substitution += 1  
            distance_matrix[i][j] = min(insertion,deletion,substitution)  
    #print(distance_matrix)
    return distance_matrix[first_length-1][second_length-1]
#杨德武故意杀人再审案 杨德武申诉案 杨德武故意杀人申诉案
'''str1 = '杨德武故意杀人再审案'
str2 = '杨德武申诉案'
str3 = '杨德武故意杀人申诉案'
print(levenshtein(str1,str3))
print(levenshtein(str2,str3))
print(levenshtein(str1,str3)/max(len(str1),len(str3)))
print(levenshtein(str2,str3)/max(len(str2),len(str3)))'''