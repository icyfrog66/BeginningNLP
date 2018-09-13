# -*- coding: utf-8 -*-
"""
Created on Fri Aug 31 23:32:21 2018

@author: Anthony
"""

import nltk
from nltk.corpus import brown
import numpy as np
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
#len(brown.tagged_sents()) == 57340
class HMMTagger():
    def __init__(self):
        self.numPossibleTags = None
        self.wordTagProbabilities = None
        self.tagToIndex = None
        self.wordToIndex = None
        self.transitionProbabilities = None
        self.possibleTags = None
        
    def train(self, sentences = brown.tagged_sents(tagset = 'universal')):
        sentences = brown.tagged_sents(tagset = 'universal')
        possibleTags = set()
        possibleWords = set()
        for sentence in sentences:
            for word in sentence:
                possibleWords.add(ps.stem(word[0].lower()))
                possibleTags.add(word[1])
        
        wordToIndex = {}
        indexToWord = {}
        for i, word in enumerate(possibleWords):
            wordToIndex[word] = i
            indexToWord[i] = word
        numWords = len(possibleWords)
        numPossibleTags = len(possibleTags)
        tagProbabilitiesGeneral = np.zeros(numPossibleTags)
        tagToIndex = {}
        indexToTag = {}
        for i, tag in enumerate(possibleTags):
            tagToIndex[tag] = i
            indexToTag[i] = tag
        #First markov table
        transitionProbabilities = np.zeros((numPossibleTags + 1, numPossibleTags))
        #second Markov Table
        wordTagProbabilities = np.zeros((numPossibleTags, numWords))
        for sentence in sentences:
            for i in range(len(sentence) - 1):
                if i == 0:
                    transitionProbabilities[len(transitionProbabilities) - 1][tagToIndex[sentence[0][1]]] += 1
                else:
                    transitionProbabilities[tagToIndex[sentence[i][1]]][tagToIndex[sentence[i+1][1]]] += 1
                wordTagProbabilities[tagToIndex[sentence[i][1]]][wordToIndex[ps.stem(sentence[i][0].lower())]] += 1
                tagProbabilitiesGeneral[tagToIndex[sentence[i][1]]] += 1
        for i in range(len(transitionProbabilities[0])):
            transitionProbabilities[:, i] /= (transitionProbabilities[:, i].sum(axis = 0) + 10**-20)
        for i in range(len(wordTagProbabilities[0])):
            wordTagProbabilities[:, i] /= (wordTagProbabilities[:, i].sum(axis = 0) + 10**-20)
        tagProbabilitiesGeneral /= sum(tagProbabilitiesGeneral)
        self.numPossibleTags = numPossibleTags
        self.wordTagProbabilities = wordTagProbabilities
        self.tagToIndex = tagToIndex
        self.wordToIndex = wordToIndex
        self.transitionProbabilities = transitionProbabilities
        self.possibleTags = possibleTags
        self.possibleWords = possibleWords

#Deal with case of unknown word later
    def predict(self, curSent = 'I am a person'):
        if not ' ' in curSent:
            return
        arrayOfWords = curSent.split(' ')
        for i in range(len(arrayOfWords)):
            arrayOfWords[i] = ps.stem(arrayOfWords[i].lower())
        arrayOfProbs = np.zeros((self.numPossibleTags, len(arrayOfWords)))
        listOfTransitionTags = []
        for i in range(self.numPossibleTags):
            cur = [0] * len(arrayOfWords)
            listOfTransitionTags.append(cur)
        for i, word in enumerate(arrayOfWords):
            if i == 0:
                for tag in self.possibleTags:
                    if self.wordTagProbabilities[self.tagToIndex[tag]][self.wordToIndex[word]] != 0:
                        arrayOfProbs[self.tagToIndex[tag]][i] = \
                            self.transitionProbabilities[len(self.transitionProbabilities) - 1][self.tagToIndex[tag]] *\
                            self.wordTagProbabilities[self.tagToIndex[tag]][self.wordToIndex[word]]
                        listOfTransitionTags[self.tagToIndex[tag]][i] = tag
            elif word not in self.possibleWords:
                print('a word in the sentence was not recognized')
                return
            else:
                for prevTag in self.possibleTags:
                    if arrayOfProbs[self.tagToIndex[prevTag]][i-1] != 0:
                        for tag in self.possibleTags:
                            probTrans = self.transitionProbabilities[self.tagToIndex[prevTag]][self.tagToIndex[tag]] * \
                                        arrayOfProbs[self.tagToIndex[prevTag]][i-1] * \
                                        self.wordTagProbabilities[self.tagToIndex[tag]][self.wordToIndex[word]] 
                            if probTrans > arrayOfProbs[self.tagToIndex[tag]][i]:
                                arrayOfProbs[self.tagToIndex[tag]][i] = probTrans
                                listOfTransitionTags[self.tagToIndex[tag]][i] = \
                                    listOfTransitionTags[self.tagToIndex[prevTag]][i-1] + ' ' + tag
        #Final result is printed
        print(listOfTransitionTags[np.argmax(arrayOfProbs[:, len(arrayOfWords) - 1])][len(arrayOfWords) - 1])