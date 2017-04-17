__author__ = 'xxxh'

#https://classroom.udacity.com/courses/ud730/lessons/6370362152/concepts/71235296110923
#Task:
#       Add 0.000 001 1million times to 1billion, then subract 1billion and check the result.

sum = 1e9

for i in range( 1000000 ):
    sum = sum + 1e-6

sum = sum - 1e9

print( sum )