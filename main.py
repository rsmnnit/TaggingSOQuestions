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
get Dataset
"""
t0 = time()
dataset = ld.getDataset(qId,questions,freq_tags)
print("time to get Dataset %0.3fs " %(time()-t0))


