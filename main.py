# -*- coding: utf-8 -*-
"""
Created on Mon Mar 26 23:59:50 2018

@author: Radhe Shyam Lodhi
"""

import load_data as ld

"""
read dataset
"""

tags, questions = ld.loadData()

"""
get 100 most frequent tags
"""
freq_tags = ld.getFrequentTag(tags)


qId = ld.getQid(freq_tags,tags)


"""
create training set
"""
training_set = ld.getTrainingset(qId,questions)


"""
create feature vector for body
"""

vocabulary,feature_vec = ld.getFeatureVector(training_set)
    

"""
calculate tfidf
"""

tfidf_body = ld.getTfidf(feature_vec)
