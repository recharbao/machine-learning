#-*- coding: utf-8 -*-
import sys
reload(sys)
sys.setdefaultencoding("utf-8")

'''
print(3 in [1,2,3,4])
a = [2,3,1,2]
print(a.sort())

s = [3, 4, 5, 6, 7, 9, 11, 13, 15, 17]
print(s[3:7])

dict = {'a' : 1, 'b' : 2}
print(dict)
print(dict.keys())
print(dict.values())


a = ['name','age','sex']
b = ['Dong',38,'Male']

c = dict(zip(a,b))
print(c)


a = [1, 2, 3, 4, 5, 6, 7]
b = a[::3]
print(b)

print([5 for i in range(10)])
'''
'''
from collections import Counter
import random
list1 = [random.randint(0,100) for i in range(1000)]
result = Counter(list1)
print(result)
'''
'''
import random
x = [random.randint(0,100) for i in range(20)]
a = sorted(x[:10])
b = sorted(x[10:20], reverse=True)
x = a + b
print(x)
'''
'''
x = input("input a list:")
i, j = input("input two cor:")
print(x[i:j + 1])
'''
'''
x = {4:'d',5:'r',7:'0'}
m = input("input a key:")
print(x.get(m, '您输入的键不存在!'))
'''

'''
import numpy as np
data = [1, 2, 3, 3]
line = [2, 3, 4, 4]

def Distance(dataOneLine, center):
    return np.sqrt(np.sum((np.array(dataOneLine) - np.array(center)) ** 2))

print(Distance(data, line))
print(min(data))

datal = [i for i in range(len(data)) if data[i] == 3]
print(datal)

p = np.array([1, 2, 3, 5, 6])

x = [[1, 2], [6, 3]]
print(np.mean(x, axis=0))



#0 : up - down
#1 : left - right

print([2, 3] in  x)

for i in enumerate(p):
    print i

x = np.arange(-2, 3)

print (np.flatnonzero(np.array([False ,True, False, False, False])))

print(p/2.0)

t = np.random.choice(p, 5, replace= False)
print(t)


x = np.array([[1, 2], [3, 5],[6, 9]])

print(x[[1, 2]])

'''
'''
import numpy as np

x = np.random.randint(1, 1000, 50)

x = x[x % 2 == 0]

print(x)
'''
'''
import random
x = [random.randint(0,100) for i in range(50)]
i = 0
while i < len(x):
    if x[i] % 2 == 1:
        del x[i]
    else:
        i = i + 1 
print(x)
'''

'''
import random
list = [random.randint(0,100) for i in range(20)]
print(list)
for i in range(0,19,2):
    for j in range(i,19,2):
        if list[i] < list[j]:
            list[i],list[j] = list[j],list[i]
print(list)
'''

'''
x = input('Please input an integer :')
t = x
i = 2
result = []
while True:
    if t==1:
        break
    if t%i==0:
        result.append(i)
        t = t/i
    else:
        i+=1
print x,'=','*'.join(map(str,result))

'''
'''
r_1 = 0
for i in range(1,100,2):
 r_1 += i
print r_1
 
r_2 = 0
for i in range(101):
 r_2 += i
print (r_2 - 50) / 2
'''
'''
import math
def _finde(x,y):

    n = x.find(y)
    if n==-1:
        return False
    else :
        n = x[n+1::].find(y)
        if n==-1:
            return True
    return False
def isPrime(x):

    for i in range(2,int(math.sqrt(x)+1)):
        if x%i == 0:
            return False
    return True


for i in range(10000):
    if isPrime(i):
        i=str(i)
        if _finde(i,"1") & _finde(i,"2") & _finde(i,"3") & _finde(i,"4"):
            print(i)
    continue

​def demo(v):

    capital = little = digit = other =0
    for i in v:
        if 'A'<=i<='Z':
            capital+=1
        elif 'a'<=i<='z':
            little+=1
        elif '0'<=i<='9':
            digit+=1
        else:
            other+=1
    return (capital,little,digit,other)
x = 'capital = little = digit = other =0'
print(demo(x))




#coding=utf-8
l = list()
while True:
    try:
        num = int(raw_input())
        l.append(num)
    except:
        break
print max(l), sum(l)



def Sum(v):
    s = 0
    for i in v:
        s += i
    return s
x = [1,2,3,4,5]
print(Sum(x))
x = (1,2,3,4,5)
print(Sum(x))





def sorted(itera):
    new_itera = []
    while itera:
        min_value = min(itera)
        new_itera.append(min_value)
        itera.remove(min_value)
　　return new_itera

'''
'''

# Python Program to create
# a data type object
import numpy as np
 
# Integer datatype
# guessed by Numpy
x = np.array([1, 2])  
print("Integer Datatype: ")
print(x.dtype)         
 
# Float datatype
# guessed by Numpy
x = np.array([1.0, 2.0]) 
print("\nFloat Datatype: ")
print(x.dtype)  
 
# Forced Datatype
x = np.array([1.2, 2.0], dtype = np.int64)   
print("\nForcing a Datatype: ")
print(x.dtype)
print(x)
'''

'''
# Python Program to create
# a data type object
import numpy as np
 
# First Array
arr1 = np.array([[4, 7], [2, 6]], 
                 dtype = np.float64)
                  
# Second Array
arr2 = np.array([[3, 6], [2, 8]], 
                 dtype = np.float64) 
 
# Addition of two Arrays
Sum = np.add(arr1, arr2)
print("Addition of Two Arrays: ")
print(Sum)
 
# Addition of all Array elements
# using predefined sum method
Sum1 = np.sum(arr1)
print("\nAddition of Array elements: ")
print(Sum1)
 
# Square root of Array
Sqrt = np.sqrt(arr1)
print("\nSquare root of Array1 elements: ")
print(Sqrt)
 
# Transpose of Array
# using In-built function 'T'
Trans_arr = arr1.T
print("\nTranspose of Array: ")
print(Trans_arr)
'''

''
s = ['eqwe','342', '4', '4']
d = set(s)
print(d)
print(set('Python'))
