# -*- coding: utf-8 -*-
"""
Created on Wed Aug 29 15:42:15 2018

@author: Anthony
"""

#Multiplying two words as vectors: if the vectors are similar, 
#the dot product, when plugged into sigmoid, will be more favorable
#for a higher percent of predicting the class of 1 instead of 0
import re
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
import numpy as np
import math
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
ps = PorterStemmer()
doc1 = open('Data/negReviews.txt').read().lower()
doc2 = open('Data/posReviews.txt').read().lower()
doc3 = open('Data/reviewTest.txt').read().lower()
doc4 = open('Data/Aligns.txt').read().lower()
text = doc1 + doc2 + doc3 + doc4
text = re.sub('\n', ' ', text)
curDoc = word_tokenize(text)
curDocs = [ps.stem(word) for word in curDoc if not word in set(stopwords.words('english'))]
text = ' '.join(curDocs)
#Creating the words, etc to use the frequencies, index to word
words = word_tokenize(text)
setOfWords = set(words)
numWords = len(setOfWords)
wordCounts = np.zeros((numWords, 1))
wordToIndex = {}
indexToWord = {}
for i, word in enumerate(setOfWords):
    wordToIndex[word] = i
    indexToWord[i] = word
#Up to this section, takes around 1 minute to run
for word in words:
    if word in wordToIndex.keys():
        wordCounts[wordToIndex[word], 0] += 1
probabilities = wordCounts / sum(wordCounts)
probabilities = probabilities.flatten()
k = 2
wordWindow = 2
numDimensions = 150
learningRate = 0.01
#The two main matrices to be used
targetWeights = 1.0/numDimensions * np.random.randn(numWords, numDimensions)
contextWeights = 1.0/numDimensions * np.random.randn(numDimensions, numWords)
sentences = sent_tokenize(text)
def sigmoid(gamma):
  if gamma < 0:
    return 1.0 - 1.0/(1.0 + math.exp(gamma))
  else:
    return 1.0/(1.0 + math.exp(-gamma))
for sentence in sentences:
    sentenceWords = word_tokenize(sentence)
    for i, word in enumerate(sentenceWords):
        #train correct ones
        mainWordIndex = wordToIndex[word]
        currentWords = []
        for j in range(max(0, i - 2), min(len(sentenceWords), i + 2)):
            if j != i:
                currentWord = sentenceWords[j]
                if currentWord in wordToIndex.keys() and word in wordToIndex.keys():
                    currentWords.append(currentWord)
                    otherWordIndex = wordToIndex[currentWord]
                    #print(np.dot(targetWeights[mainWordIndex, :], contextWeights[:, otherWordIndex]))
                    #sigmoidProbability = 1.0/(1.0 + math.exp(-1.0 * np.dot
                        #(targetWeights[mainWordIndex, :], contextWeights[:, otherWordIndex])))
                    sigmoidProbability = sigmoid(np.dot(targetWeights[mainWordIndex, :], 
                                                        contextWeights[:, otherWordIndex]))
                    targetWeights[mainWordIndex, :] -= learningRate * targetWeights[mainWordIndex, :] \
                        * (sigmoidProbability - 1.0)
                    contextWeights[:, otherWordIndex] -= learningRate * contextWeights[:, otherWordIndex] \
                        * (sigmoidProbability - 1.0)
        #train incorrect ones
        for j in range(0, wordWindow * k):
            index = np.random.choice(len(probabilities), 1, p = probabilities)[0]
            while indexToWord[index] in currentWords:
                index = np.random.choice(len(probabilities), 1, p = probabilities)[0]
            otherWord = indexToWord[index]
            if otherWord in wordToIndex.keys() and word in wordToIndex.keys():
                sigmoidProbability = sigmoid(np.dot(targetWeights[mainWordIndex, :], 
                                                    contextWeights[:, index]))
                targetWeights[mainWordIndex, :] -= learningRate * targetWeights[mainWordIndex, :] \
                    * sigmoidProbability
                contextWeights[:, index] -= learningRate * contextWeights[:, index] \
                    * sigmoidProbability
