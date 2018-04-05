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

"""
read dataset
"""


#tags, questions = ld.loadData()


#tags_train = tags.sample(frac = 0.3)
#tags_test = tags.loc[~tags.index.isin(tags_train.index)]



"""
get 40 most frequent tags
"""
#freq_tags = ld.getFrequentTag(tags_train)
#print (freq_tags)

positive,negative = ld.getQid2(tags_train,'javascript')
 


train_dataset, test_dataset = ld.getQuestions(positive,negative,questions)

train_dataset.to_csv('js_train.csv')
test_dataset.to_csv('js_test.csv')


X_train = train_dataset['Body']
Y_train = train_dataset['Class']
#X_train = X_train['Body']

X_test = test_dataset['Body']
Y_test = test_dataset['Class']
#X_test = X_test['Body']


#print ("dataset")

#print (X_train.shape,Y_train.shape)

#print ("dataset over")


vectorizer = TfidfVectorizer(stop_words='english')

X_train_idf = vectorizer.fit_transform(X_train).toarray()
X_test_idf = vectorizer.transform(X_test).toarray()

naiveBayes = GaussianNB()

naiveBayes.fit(X_train_idf,Y_train.values)
y_pred = naiveBayes.predict(X_test_idf)
print (naiveBayes.score(X_test_idf,Y_test.values))
               

print (confusion_matrix(Y_test,y_pred))

print (precision_recall_fscore_support(Y_test,y_pred))

#qId = ld.getQid(freq_tags,tags)


"""
get Dataset
"""
#dataset = ld.getDataset(qId,questions,freq_tags)
#dataset.to_csv("dataset.csv")
"""
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

"""



"""
Naive Bayes Classifier
"""


"""
naiveBayes = GaussianNB()
"""


"""
Get training and test dataset
"""

"""
naiveBayes.fit(X_train_idf,Y_train[:,97:98].flatten())
y_pred = naiveBayes.predict(X_test_idf)
#print (naiveBayes.score(X_test_idf,Y_test[:,1:2].flatten()))
               


print (confusion_matrix(Y_test[:,97:98],y_pred))

print (precision_recall_fscore_support(Y_test[:,97:98],y_pred))
"""


