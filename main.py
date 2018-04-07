# -*- coding: utf-8 -*-
"""
Created on Mon Mar 26 23:59:50 2018

@author: Radhe Shyam Lodhi
"""

import load_data as ld
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from skmultilearn.problem_transform import BinaryRelevance
from skmultilearn.problem_transform import ClassifierChain
from sklearn.metrics import accuracy_score

"""
read dataset
"""
#tags, questions = ld.loadData()



"""
get 100 most frequent tags
"""
#freq_tags = ld.getFrequentTag(tags)




#qId = ld.getQid(freq_tags,tags)


"""
get Dataset
"""
#dataset = ld.getDataset(qId,questions,freq_tags)
#dataset.to_csv("dataset.csv")

dataset = pd.read_csv("./dataset.csv",engine='python')

vectorizer = TfidfVectorizer(stop_words='english')

train = dataset.sample(frac=0.5)
test = dataset.loc[~dataset.index.isin(train.index)]

X_train = train['Body']+train['Title']
X_train_idf = vectorizer.fit_transform(X_train).toarray()
X_test = test['Body'] + test['Title']
X_test_idf = vectorizer.transform(X_test).toarray()

Y_train = train.iloc[:,4:].values
Y_test = test.iloc[:,4:].values

print (Y_test)




"""
Naive Bayes Classifier
"""
#naiveBayes = GaussianNB()

classifier = ClassifierChain(GaussianNB())
classifier.fit(X_train_idf,Y_train)
predictions = classifier.predict(X_test_idf)
print (accuracy_score(Y_test,predictions))


"""
Get training and test dataset
"""

"""
naiveBayes.fit(X_train_idf,Y_train[:,97:98].flatten())
y_pred = naiveBayes.predict(X_test_idf)
"""



#print (naiveBayes.score(X_test_idf,Y_test[:,1:2].flatten()))
               

"""
print (confusion_matrix(Y_test[:,97:98],y_pred))

print (precision_recall_fscore_support(Y_test[:,97:98],y_pred))
"""


