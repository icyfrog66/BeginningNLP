# -*- coding: utf-8 -*-
"""
Created on Wed Aug 29 14:53:15 2018

@author: Anthony
"""
import re
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
import numpy as np
import math
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
ps = PorterStemmer()
doc1 = open('Data/negReviews.txt').read().lower()
doc1 = re.sub('\n', ' ', doc1)
doc2 = open('Data/posReviews.txt').read().lower()
doc2 = re.sub('\n', ' ', doc2)
doc3 = open('Data/reviewTest.txt').read().lower()
doc3 = re.sub('\n', ' ', doc3)
doc4 = open('Data/Aligns.txt').read().lower()
doc4 = re.sub('\n', ' ', doc4)
docs = []
docs.append(doc1)
docs.append(doc2)
docs.append(doc3)
docs.append(doc4)
#This implementation removes stopwords, takes around 20 seconds
for i, doc in enumerate(docs):
    curDoc = word_tokenize(doc)
    curDocs = [ps.stem(word) for word in curDoc if not word in set(stopwords.words('english'))]
    curDoc = ' '.join(curDocs)
    docs[i] = curDoc
fullDocs = ''
for doc in docs:
    fullDocs += doc
setOfWords = set(word_tokenize(fullDocs))
numWords = len(setOfWords)
wordToWord = np.zeros((numWords, numWords))
docFrequency = np.zeros((numWords, 1))
wordToIndex = {}
indexToWord = {}
numDocs = 4
#Set the dictionaries for word to index and index to word
for i, word in enumerate(setOfWords):
    wordToIndex[word] = i
    indexToWord[i] = word
#Set the number of documents each word appears in
for doc in docs:
    docWordSet = set(word_tokenize(doc))
    for word in docWordSet:
        if word in wordToIndex.keys():
            docFrequency[wordToIndex[word]] += 1
#Number of words forward or backwards to look
constant = 4
for doc in docs:
    sentences = sent_tokenize(doc)
    for sentence in sentences:
        words = word_tokenize(sentence)
        for i, word in enumerate(words):
            for j in range(max(i - 4, 0), min(i + 4, len(words))):
                if j != i:
                    if words[i] in wordToIndex.keys() and words[j] in wordToIndex.keys():
                        wordToWord[wordToIndex[words[i]]][wordToIndex[words[j]]] += 1
for i in range(len(docFrequency)):
    if docFrequency[i] != 0:
        wordToWord[i, :] *= math.log(numDocs/docFrequency[i])


#Write functions for closest words, word distance (cosine distance)