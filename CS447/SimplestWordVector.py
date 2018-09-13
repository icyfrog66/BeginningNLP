# -*- coding: utf-8 -*-
"""
Created on Fri Aug  3 22:35:42 2018

@author: Anthony
"""
import nltk, re
import numpy as np
from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt
#Based on Lecture 2: Beginning, introduction methodology
data = open(file = 'Data/Instinctively.txt').read()
data = re.sub("[^a-zA-Z\.']"," ", data)
#File now has no punctuation, making tokenizer cleaner
sentences = nltk.sent_tokenize(data)
for i in range(len(sentences)):
    sentences[i] = nltk.word_tokenize(sentences[i])
#Word maps to number, array index
WordIndexDict = {}
#Reverse of previous dict: used to label words on plot
ReverseWordIndexDict = {}
counter = 0
#Fill dictionary with possible words
for sentence in sentences:
    for word in sentence:
        if word not in WordIndexDict.keys():
            WordIndexDict[word] = counter
            ReverseWordIndexDict[counter] = word
            counter += 1

unEncodedArray = np.zeros((counter, counter))

#Encoding with the skip gram: plus or minus 4 words
for sentence in sentences:
    for i in range(len(sentence)): 
        word = sentence[i]
        for j in range(max(0, i - 4), min(len(sentence), i + 4)):
            if j != i:
                otherWord = sentence[j]
                unEncodedArray[WordIndexDict[word]][WordIndexDict[otherWord]] += 1
#Too slow
#U, sigma, V = np.linalg.svd(unEncodedArray, full_matrices = False)
#TruncatedSVD is used instead: takes 20 seconds
svd = TruncatedSVD(n_components = 50)
encoded = svd.fit_transform(unEncodedArray)
plt.scatter(encoded[:, 0], encoded[:, 1])
#Plot is slow to load once the labels are added
for i in range(counter):
    plt.annotate(s = ReverseWordIndexDict[i], xy = (encoded[i, 0], encoded[i, 1]))
    