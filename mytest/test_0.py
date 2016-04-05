__author__ = 'Thong_Le'

from numpy import genfromtxt

my_data = genfromtxt('a.csv', delimiter='\t')

print(my_data)