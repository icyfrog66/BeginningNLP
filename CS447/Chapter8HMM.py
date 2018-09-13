# -*- coding: utf-8 -*-
"""
Created on Wed Aug 29 17:09:08 2018

@author: Anthony
"""

#Markov: Q is a set of N states, A is a transition probability matrix 
#with probabilities of going from state i to j, and pi is an 
#initial probability distribution (change of initial state at each state)

#Hidden model: Q, A, O is a set of T observations drawn from vocab V, 
#B is probability of observation o_t being generated from state i, 
#still have pi as the initial state probabilities. b is called emission probs.

#A: probability of a certain speech tag occuring after a speech tag.

#First HMM: Will just do if else cases when a word is unknown, lol!!!

import nltk
from nltk.corpus import brown
import numpy as np
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
#len(brown.tagged_sents()) == 57340
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

#Deal with case of unknown word later
curSent = 'I do not like being a person'
arrayOfWords = curSent.split(' ')
for i in range(len(arrayOfWords)):
    arrayOfWords[i] = ps.stem(arrayOfWords[i].lower())
arrayOfProbs = np.zeros((numPossibleTags, len(arrayOfWords)))
listOfTransitionTags = []
for i in range(numPossibleTags):
    cur = [0] * len(arrayOfWords)
    listOfTransitionTags.append(cur)
for i, word in enumerate(arrayOfWords):
    if i == 0:
        for tag in possibleTags:
            if wordTagProbabilities[tagToIndex[tag]][wordToIndex[word]] != 0:
                arrayOfProbs[tagToIndex[tag]][i] = \
                    transitionProbabilities[len(transitionProbabilities) - 1][tagToIndex[tag]] *\
                    wordTagProbabilities[tagToIndex[tag]][wordToIndex[word]]
                listOfTransitionTags[tagToIndex[tag]][i] = tag
    elif word not in possibleWords:
        print('a word in the sentence was not recognized')
        break
    else:
        for prevTag in possibleTags:
            if arrayOfProbs[tagToIndex[prevTag]][i-1] != 0:
                for tag in possibleTags:
                    probTrans = transitionProbabilities[tagToIndex[prevTag]][tagToIndex[tag]] * \
                                arrayOfProbs[tagToIndex[prevTag]][i-1] * \
                                wordTagProbabilities[tagToIndex[tag]][wordToIndex[word]] 
                    print(str(probTrans))
                    if probTrans > arrayOfProbs[tagToIndex[tag]][i]:
                        arrayOfProbs[tagToIndex[tag]][i] = probTrans
                        listOfTransitionTags[tagToIndex[tag]][i] = \
                            listOfTransitionTags[tagToIndex[prevTag]][i-1] + ' ' + tag
#Final result is printed
print(listOfTransitionTags[np.argmax(arrayOfProbs[:, len(arrayOfWords) - 1])][len(arrayOfWords) - 1])