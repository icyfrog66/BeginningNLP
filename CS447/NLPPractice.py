# Natural Language Processing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Data/Aligns.txt', delimiter = '\t', quoting = 3)

# Cleaning the texts
import re
import nltk
#Remove words that are usually irrelevant: these include things like this, that, the, [prepositions]
#nltk.download('stopwords')
#Inporting stopwords to get stopwords have a name
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
for i in range(0, len(dataset)):
    #Input what not to remove: don't remove a-z, A-Z. Other characters like punctuation are removed.
    #Third argument is the place to remove. The ith review is indexed. Removed characters are a space.
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
    #All to lower case
    review = review.lower()
    #Split review into different words
    review = review.split()
    #Used to get stems, as only word stems are important for sentiment (usually)
    ps = PorterStemmer()
    #Iterate through words in review, words not in stopwords are the only ones checked
    #Set of stopwords is very useful for longer texts
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    #Words are separated right now, so words need to be joined back to a word
    review = ' '.join(review)
    corpus.append(review)

# Creating the Bag of Words model: Sparse matrix to be used in classification
from sklearn.feature_extraction.text import CountVectorizer
#CountVectorizer has many things that can be useful, such as converting to lowercase
#Keeps only most common words
cv = CountVectorizer(max_features = 1500)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, 1].values

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Fitting Naive Bayes to the Training set
"""from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)"""

from sklearn.svm import SVR
classifier = SVR(kernel = 'rbf')
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred1 = classifier.predict(X_test)
#Round numpy if necessary
y_pred = np.round(y_pred1, decimals = 0)
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
jj = 0
kk = 0
ll = 0
for i in range(0, len(cm)):
    for j in range(0, len(cm)):
        kk = kk + cm[i][j]
        if i == j:
            jj = jj + cm[i][j]
        if abs(i-j)<=1:
            ll = ll + cm[i][j]


"""
141: 34.48%
184: 37.84%
197: 45% (might have added a few longer reviews in this iteration)
216: 43.18% (adding almost only long reviews)
269: 29.63%
(Reversion to 180: 36.11%)
(Last 167 of 269: 41.18%)
(Somehow, moving them in a different order, 1st half to end, gives 53.70% in 269)
Thus, the question is, is some data better for the training set, and some better for test?

SVR with 269: 33.33% with exact values (on rounding), 81.48% on rounding to integers
"""