# -*- coding: utf-8 -*-
"""
Created on Mon Mar 26 23:59:50 2018

@author: Radhe Shyam Lodhi
"""

import load_data as ld
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import GaussianNB

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


vectorizer = TfidfVectorizer(stop_words='english')

train = dataset.sample(frac=0.7)
test = dataset.loc[~dataset.index.isin(train.index)]

X_train = train['Body']+train['Title']
X_train_idf = vectorizer.fit_transform(X_train)
X_test = test['Body'] + test['Title']
X_test_idf = vectorizer.fit(X_test)

Y_train = train.iloc[:,3:]
Y_test = test.iloc[:,3:]




"""
Naive Bayes Classifier
"""
naiveBayes = GaussianNB()

"""
Get training and test dataset
"""

naiveBayes.fit(X_train_idf,Y_train.iloc[:,1:2].toarray())








