# -*- coding: utf-8 -*-
"""
Created on Sun Aug 26 18:31:46 2018

@author: Anthony
"""

import re
import nltk
import math
import numpy as np
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import sent_tokenize, word_tokenize
ps = PorterStemmer()
negative = open('Data/negReviews.txt').read().lower().split('\n')
positive = open('Data/posReviews.txt').read().lower().split('\n')
unknown = open('Data/reviewTest.txt').read().lower().split('\n')
setOfWords = set()
overallCorpus = negative + positive + unknown
overallCorpus = ' '.join(overallCorpus)
overallCorpus = word_tokenize(overallCorpus)
overallCorpus = [ps.stem(word) for word in overallCorpus if not word in set(stopwords.words('english'))]
wordToIndex = {}
indexToWord = {}
for word in overallCorpus:
    setOfWords.add(word)
for i, word in enumerate(setOfWords):
    wordToIndex[word] = i
    indexToWord[i] = word
reviews = []
for review in negative:
    review = re.sub('[^a-z]', ' ', review)
    review = word_tokenize(review)
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    currentEntry = [review, 0]
    reviews.append(currentEntry)
for review in positive:
    review = re.sub('[^a-z]', ' ', review)
    review = word_tokenize(review)
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    currentEntry = [review, 1]
    reviews.append(currentEntry)
reviews = np.array(reviews)
countArray = np.zeros((reviews.shape[0], len(setOfWords)))
for i in range(reviews.shape[0]):
    rev = reviews[i, :]
    words = word_tokenize(rev[0])
    category = rev[1]
    for word in words:
        word = ps.stem(word)
        if word in wordToIndex.keys():
            countArray[i][wordToIndex[word]] += 1
countArray += 1
negProb = math.log(float(len(negative)/(len(negative)+len(positive))))
posProb = math.log(float(len(positive)/(len(negative)+len(positive))))
logProbs = np.zeros((countArray.shape[1], 2))
negSum = np.sum(countArray[0:len(negative), :])
posSum = np.sum(countArray[len(negative):, :])
for i in range(countArray.shape[1]):
    numNegative = sum(countArray[0:len(negative), i])
    numPositive = sum(countArray[len(negative):, i])
    logProbs[i][0] = math.log(numNegative/negSum)
    logProbs[i][1] = math.log(numPositive/posSum)

def predict(string):
    string = word_tokenize(string)
    string = [ps.stem(word) for word in string if not word in set(stopwords.words('english'))]
    string = ' '.join(string)
    string = word_tokenize(string)
    negVal = 0
    posVal = 0
    for word in string:
        if word in wordToIndex.keys():
            negVal += logProbs[wordToIndex[word]][0]
            posVal += logProbs[wordToIndex[word]][1]
    negVal = math.exp(negVal) * math.exp(negProb)
    posVal = math.exp(posVal) * math.exp(posProb)
    if negVal > posVal:
        print('Negative: ' + str(negVal))
    else:
        print('Positive: ' + str(posVal))
predict('hate')
predict('love love align pant')
predict('piling')