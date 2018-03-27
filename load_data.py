# -*- coding: utf-8 -*-
"""
Created on Mon Mar 26 19:42:08 2018

@author: Radhe Shyam Lodhi
"""


import pandas as pd
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer



"""
read dataset
"""

def loadData():
    tags = pd.read_csv('./Tags.csv',engine = 'python')
    questions = pd.read_csv('./Questions.csv', engine='python')


    """
    drop unused fields
    """
    #tag = tags.drop(['Tag'],axis = 1)
    question = questions.drop(['OwnerUserId','CreationDate','ClosedDate','Score'],axis=1)
    
    return tags,question

"""
take 100 most frequent tags
"""

def getFrequentTag(tags):
    tag = tags['Tag'].values.tolist()
    counts = Counter(tag).most_common(100)

    return counts



def getQid(freq_tags,tags):
    qid = []
    for p in freq_tags:
        cnt = 100
        for q in tags.itertuples():
            if p[0] == q[2]:
                qid.append((q[1],q[2]))
                cnt-=1
            if cnt == 0:
                break
    return qid
                


"""
create training set
"""

def getTrainingset(qId,ques):
    training_set = []
    i = 0
    while (i<len(qId)):
        df1 = ques[(ques['Id']==qId[i][0])]
        i = i+1
        for q in df1.itertuples():
            training_set.append((q[1],q[2],q[3]))
    return training_set





"""
create feature vector for body
"""

def getFeatureVector(training_set):
    #,ngram_range=(1, 2)
    vectorizer = CountVectorizer(stop_words='english',token_pattern=r'\b\w+\b', min_df=1)
    body = []
    for c,d,e in training_set:
        body.append(e)

    X = vectorizer.fit_transform(body)
    #analyze = vectorizer.build_analyzer()

    featureVectorBody = X.toarray() 

    return vectorizer.vocabulary_,featureVectorBody






"""
calculate tfidf
"""

def getTfidf(feature_vec):
    transformer = TfidfTransformer(smooth_idf=False)
    tfidf = transformer.fit_transform(feature_vec)
    return tfidf.toarray()
    



