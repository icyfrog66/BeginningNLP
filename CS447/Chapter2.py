# -*- coding: utf-8 -*-
"""
Created on Sat Aug 25 14:25:52 2018

@author: Anthony
"""

#Mostly starting in section 2.4
import pandas as pd
file = pd.read_csv('Data/SAOJP1.txt')
import tinysegmenter
data = open('Data/SAOJP1.txt', encoding = 'utf-8').read()
segmenter = tinysegmenter.TinySegmenter()
print(' | '.join(segmenter.tokenize(data[:1000])))



#Not sure about this part: reading the dictionary files for words
dicto = open('Data/edict2u', encoding = 'utf-8')
array = []
for line in dicto:
    array.append(line)
dictionary = {}
for entry in array:
    dictionary[entry[:entry.find(';')]] = entry[entry.find('[') + 1: entry.find(']')]
