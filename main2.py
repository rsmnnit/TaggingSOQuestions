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


tags, questions = ld.loadData()



"""
get 3 most frequent tags
"""
freq_tags = ld.getFrequentTag(tags_train)
print (freq_tags)


questions_train = questions.sample(frac = 0.05)
questions_train_ids = questions_train['Id'].values
questions_test = questions.loc[~questions.index.isin(questions_train.index)]
questions_test_ids = questions_test['Id'].values


vectorizer2 = TfidfVectorizer(stop_words='english')
question_test_content = questions_test['Title'] + questions_test['Body']
question_test_tfidf = vectorizer2.transform(question_test_content).toarray()



true_df = pd.DataFrane(columns = freq_tags)
i=0

for question_id in questions_test_ids:
    temp = []
    relevant_tags = tags[(tags['Id']==question_id)]['Tag'].values
    for ind_tag in freq_tags:
        if ind_tag in relevant_tags:
            temp.append(1)
        else:
            temp.append(0)
    true_df.loc[i] = temp
    i=i+1
    
Y_test = true_df.values
            


tags_train = tags[(tags['Id'].isin(questions_train_ids))]
tags_test = tags.loc[~tags.index.isin(tags_train.index)]






test_df = pd.DataFrame()



for tags in freq_tags.itertuples():
    tag = tags[1]
    positive,negative = ld.getQid2(tags_train,tag)

    train_dataset, test_dataset = ld.getQuestions(positive,negative,questions_train)

    train_dataset.to_csv('./tagsdataset/'+tag+'_train.csv')
    test_dataset.to_csv('./tagsdataset/'+tag+'_test.csv')

    
    vectorizer = TfidfVectorizer(stop_words='english')
    
    X_train = train_dataset['Body']
    X_train_idf = vectorizer.fit_transform(X_train).toarray()
     
     
    Y_train = train_dataset['Class']

    #X_test = questions_test['Body']
    #X_test_idf = vectorizer.transform(X_test).toarray()
    #Y_test = questions_test['Class']
    
    
    naiveBayes = GaussianNB()
    naiveBayes.fit(X_train_idf,Y_train.values)
    
    y_pred = naiveBayes.predict(question_test_tfidf)
    
    test_df[tag] = y_pred
    
    



"""
vectorizer = TfidfVectorizer(stop_words='english')

X_train_idf = vectorizer.fit_transform(X_train).toarray()
X_test_idf = vectorizer.transform(X_test).toarray()

naiveBayes = GaussianNB()

naiveBayes.fit(X_train_idf,Y_train.values)
y_pred = naiveBayes.predict(X_test_idf)
print (naiveBayes.score(X_test_idf,Y_test.values))
               

print (confusion_matrix(Y_test,y_pred))

print (precision_recall_fscore_support(Y_test,y_pred))
"""

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


