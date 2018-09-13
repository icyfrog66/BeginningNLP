# -*- coding: utf-8 -*-
"""
Created on Mon Aug 27 11:33:31 2018

@author: Anthony
"""
import nltk
from nltk import freqdist
import numpy as np
import math
from nltk.stem.porter import PorterStemmer

negative = open('Data/negReviews.txt').read().lower().split('\n')
positive = open('Data/posReviews.txt').read().lower().split('\n')
unknown = open('Data/reviewTest.txt').read().lower().split('\n')

ps = PorterStemmer()
#List of Feature Words: leg, pant, pill, love, wear, like, dissapoint, align, 
#feel, qualiti, comfort, fabric, first, bought, soft, howev, lululemon,
#seam, want, alreadi, perfect, absolut, recommend, comfi, pile, 25 features total
featuredWords = ['leg', 'pant', 'pill', 'love', 'wear', 'like', 
                 'dissapoint', 'aling', 'feel', 'qualiti', 'comfort',
                 'fabric', 'first', 'bought', 'soft', 'howev', 'lululemon', 
                 'seam', 'want', 'alreadi', 'perfect', 'absolut', 'recommend', 
                 'comfi', 'pile']
weights = np.random.normal(size = len(featuredWords) + 1)
#bias is the last value

def preprocess(string):
    string = word_tokenize(string)
    string = [ps.stem(word) for word in string if not word in set(stopwords.words('english'))]
    string = ' '.join(string)
    return string

def predict(string):
    counts = []
    string = preprocess(string)
    for word in featuredWords:
        counts.append(string.split().count(word))
    counts.append(1)
    counts = np.array(counts)
    prediction = 1.0/(1 + math.exp(-1 * np.dot(weights, counts)))
    print(prediction)
    if prediction >= 0.5:
        print('positive')
    else:
        print('negative')


learningRate = 0.1
regularizationL1 = 0.005
#1 is positive, 0 is negative for category
def train(review, category, weights):
    counts = []
    for word in featuredWords:
        counts.append(review.split().count(word))
    counts.append(1)
    counts = np.array(counts)
    logDerivativeConstant = 1.0/(math.exp(-1 * np.dot(weights, counts)) + 1) - float(category)
    weights = weights - learningRate * logDerivativeConstant * counts 
    return weights
#weights = train(negative[1], 0, weights)
for review in negative:
    review = preprocess(review)
    weights = train(review, 0, weights)
for review in positive:
    review = preprocess(review)
    weights = train(review, 1, weights)
for review in unknown:
    predict(review)
    
#Largely predicting positive: could be due to poorly chosen features
