# -*- coding: utf-8 -*-
"""
Created on Mon Mar 26 19:42:08 2018

@author: Radhe Shyam Lodhi
"""


import pandas as pd
import re
from collections import Counter



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
    #tag = tags['Tag'].values.tolist()
    counts = Counter(tags).most_common(40)
    freq_tags = pd.DataFrame(counts)
    freq_tags.columns = ['Tag','count']
    #freq_tags = freq_tags.drop(['count'],axis=1)
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


def getQid2(tags,tag):
    positive_whole = tags[(tags['Tag']==tag)]['Id']
    negative_whole = tags[(tags['Tag']!=tag)]['Id']
     



    positive = positive_whole.sample(frac=0.5)
    negative = negative_whole.sample(frac = 0.06)
    #complete = pd.concat([positive,negative])
    
    """
    print ("Classes")
    print (positive_whole,negative_whole)
    print ("Classes Over")
    """
    
    return positive,negative




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
                
                
                mask = dataset['Id']==qId[i][0] 
                dataset.loc[mask,qId[i][1]]=1
                
                
        #print(qId[i][0])
        i= i+1
    return dataset





def getQuestions(positive,negative,questions):
    positive_questions = []
    negative_questions = []
    
    #questions = questions.sample( frac = 0.5)
    
    for positive_id in positive:
        
        title = questions[(questions['Id']==positive_id)]['Title'].values[0]
        question = questions[(questions['Id']==positive_id)]['Body'].values[0]
        positive_questions.append((title,question,1))
        
    for negative_id in negative:
        title = questions[(questions['Id']==negative_id)]['Title'].values[0]
        question = questions[(questions['Id']==negative_id)]['Body'].values[0]
        negative_questions.append((title,question,0))
        

    positive_dataset = pd.DataFrame(positive_questions)
    positive_dataset.columns = ['Title','Body','Class']
    positive_train = positive_dataset.sample(n=min(500,len(positive_dataset)))
    #positive_test = positive_dataset.loc[~positive_dataset.index.isin(positive_train.index)]  
    
    negative_dataset = pd.DataFrame(negative_questions)
    negative_dataset.columns = ['Title','Body','Class']
    negative_train = negative_dataset.sample(n=min(700,len(negative_dataset)))
    #negative_test = negative_dataset.loc[~negative_dataset.index.isin(negative_train.index)] 
      
        
    #train_dataset = pd.Dataframe(positive_train,columns = ['Title','Body','Class'])
    train_dataset = positive_train.append(negative_train)
    
    #test_dataset = pd.Dataframe(positive_test,columns = ['Title','Body','Class'])
    #test_dataset = positive_test.append(negative_test) 
    
    return train_dataset
  
  
    
    

"""
Function to remove links from question content
"""

def removeLinks(content):
    temp = content
    pattern = re.compile(r'<a href>(.*)</a>') #create a pattern
    pattern.sub('',temp) #replace the pattern with empty string
    return temp
