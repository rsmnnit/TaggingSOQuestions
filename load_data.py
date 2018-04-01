# -*- coding: utf-8 -*-
"""
Created on Mon Mar 26 19:42:08 2018

@author: Radhe Shyam Lodhi
"""


import pandas as pd
import re
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
    freq_tags = pd.DataFrame(counts)
    freq_tags.columns = ['Tag','count']
    freq_tags = freq_tags.drop(['count'],axis=1)
    return freq_tags



def getQid(freq_tags,tags):
    qid = []
    for p in freq_tags.itertuples():
        cnt = 100
        for q in tags.itertuples():
            
            if p[1] == q[2]:
                qid.append((q[1],q[2])) # Id, Tag
                cnt-=1
            if cnt == 0:
                break
    return qid            






"""
create dataset set
"""
def getDataset(qId,ques,freq_tags):
    dataset = pd.DataFrame(pd.np.empty((0, 103)))
    tagslist = freq_tags['Tag'].values.tolist()
    col = ['Id','Title','Body']
    i = 0
    
    """
    Add all frequent tags as features
    """
    while i<len(tagslist):
        col.append(tagslist[i])
        i=i+1
    
    
    dataset.columns = [col]
    i = 0
    while (i<len(qId)):
        df1 = ques[(ques['Id']==qId[i][0])]
        for q in df1.itertuples():
            df = dataset[(dataset['Id']==qId[i][0])]
            if df.empty:
                temp = []
                temp.append(q[1]) #Id
                temp.append(q[2]) # title
                temp.append(q[3]) #body
                for p in freq_tags.itertuples():
                    if qId[i][1]==p[1]: # tag of (qid or ques) == tag from list of tags
                        temp.append(1)
                    else:
                        temp.append(0)
                
                dataset = dataset.append(pd.Series(temp,index = col),ignore_index=True)
                
            else:
                
                dataset.drop(df,axis=1)
                df[qId[i][1]] = 1
                dataset.append(df)
                
                
        #print(qId[i][0])
        i= i+1
    return dataset



"""
Function to remove links from question content
"""

def removeLinks(content):
    temp = content
    pattern = re.compile(r'<a href>(.*)</a>')
    pattern.sub('',temp)
    return temp
