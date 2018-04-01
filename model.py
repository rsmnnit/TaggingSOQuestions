# -*- coding: utf-8 -*-
"""
Created on Sat Mar 17 21:47:28 2018

@author: spark
"""

import pandas as pd 

"""

Print Panda Version
print (pd.__version__)

"""

questions = pd.read_csv('./Questions.csv')
tags = pd.read_csv('./Tags.csv') 

print ("Tags Data")
print (tags.head())

print ("Questions Data")
print (questions.head())

print ("Tags Columns")
print (tags.columns.values)

print ("Question Columns")
print (questions.columns.values)


#Removing extra fields from Questions Data
questions = questions.drop(['OwnerUserId','CreationDate','ClosedDate','Score'],axis=1)

print ("Usefule Question Columns")
print (questions.columns.values)

print (tags.count())
print (questions.count())


#Print uniqu tags count
#print (tags.groupby('Tag').count().sort_values(['Tag']))