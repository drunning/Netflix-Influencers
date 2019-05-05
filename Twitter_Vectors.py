import re
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from operator import itemgetter
import preprocessor as p
from itertools import chain
from string import punctuation
import math
import pandas_profiling
import pickle
from collections import Counter
from datetime import datetime
#
df=pd.read_csv('Twitter_Only_English_database.csv', sep=',')
df=df[['id',  'content', 'new_date', 'retweets_formatted', 'favorites_formatted']]


## *************************** definitions *************************************
def removePunc(input_text):
    input_text = str(input_text)
    updated_text =input_text.strip(punctuation)
    return updated_text

pic_pattern = re.compile(r'pic.twitter.com/[0-9a-zA-Z]{10}')
url_pattern = re.compile(r'(?i)((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?\xab\xbb\u201c\u201d\u2018\u2019]))')
def cleanDataFrame(df):
    df['content_cleaned'] = df['content'].apply(lambda x: pic_pattern.sub('', x))
    df['content_cleaned'] = df['content_cleaned'].apply(lambda x: url_pattern.sub('',x))
    df['content_cleaned'] = df['content_cleaned'].apply(lambda x: ' '.join([word.lower() for word in x.split()]))
    df['content_cleaned'] = df['content_cleaned'].apply(removePunc)
    return df


def getHashtags(df):
    hash_pattern = re.compile(r'#\w*')
    df['hashtags'] = df['content_cleaned'].apply(lambda x: hash_pattern.findall(x))
    return df


def getMentions(df):
    mention_pattern = re.compile(r'@\w*')
    df['mentions'] = df['content_cleaned'].apply(lambda x: mention_pattern.findall(x))
    return df


def getHashtagSet(df_column):
    hashtag_list = df_column.tolist()
    hashtag_list = list(filter(None, hashtag_list))
    hashtag_list = list(chain.from_iterable(hashtag_list))
    all_hashtags = []
    for item in hashtag_list:
        all_hashtags.append(item.lower())
    return set(all_hashtags)


def getMentionSet(df_column):
    mention_list = df_column.tolist()
    mention_list = list(filter(None, mention_list))
    mention_list = list(chain.from_iterable(mention_list))
    all_mentions = []
    for item in mention_list:
        all_mentions.append(item.lower())
    return set(all_mentions)

#
## *************************** df cleaning *************************************

df = cleanDataFrame(df)
df = getHashtags(df)
df = getMentions(df)

#
## *************************** create mentions and hashtags ********************

mentionSet = getMentionSet(df['mentions'])
hashtagSet = getHashtagSet(df['hashtags'])


## create dicts from sets
mentionDict = {el:0 for el in list(mentionSet)}
hashtagDict = {el:0 for el in list(hashtagSet)}


## fill the dictionary with sum of retweets^2
for row in df.itertuples(index=False):
    content = row.mentions
    for item in content:
        if item in mentionDict:
            mentionDict[item] += row.retweets_formatted**2

for row in df.itertuples(index=False):
    content = row.hashtags
    for item in content:
        if item in hashtagDict:
            hashtagDict[item] += row.retweets_formatted**2


## drop certain mentions
mentionDict.pop('@netflix')
mentionDict.pop('@')
#
## drop certain hashtags
hashtagDict.pop('#netflix')
hashtagDict.pop('#rt')
hashtagDict.pop('#')


## use the dictionaries with the squared values to:
## 1. Drop any mention with <= 111200 sum squared retweets
## 2. Drop any hashtag with <= 246230 sum squared retweets


mentionsToDrop = []
hashtagsToDrop = []

for key, value in mentionDict.items():
    if value < 111200:
        mentionsToDrop.append(key)

for key, value in hashtagDict.items():
    if value < 246230:
        hashtagsToDrop.append(key)


## drop all of the mentions and hashtags noted above
for item in mentionsToDrop:
    mentionDict.pop(item)

for item in hashtagsToDrop:
    hashtagDict.pop(item)



 create dataframes from the dictionaries
mentionDictDF = pd.DataFrame(mentionDict.items(), columns=['Mention', 'SumRetweetsSquared'])
hashtagDictDF = pd.DataFrame(hashtagDict.items(), columns=['Hashtag', 'SumRetweetsSquared'])

## csv files of the new datasets
mentionDictDF.to_csv('mentions_Final.csv', sep=',', index=None)
hashtagDictDF.to_csv('hashtags_Final.csv', sep=',', index=None)

 send these final round dataframes to stats
pfr = pandas_profiling.ProfileReport(mentionDictDF)
pfr.to_file('Quick_Stats_summary_mention_Final.html')
pfr = pandas_profiling.ProfileReport(hashtagDictDF)
pfr.to_file('Quick_Stats_summary_hashtag_Final.html')


## *************************** create vectors **********************************
#
## create the vocabularies for the countvectorizer
mentionVocabulary = []
for key in mentionDict.keys():
    mentionVocabulary.append(key)

hashtagVocabulary = []
for key in hashtagDict.keys():
    hashtagVocabulary.append(key)

mentionVocabulary.extend(hashtagVocabulary)
vectorVocab = mentionVocabulary


## create the new vectors and concat with df
vec = CountVectorizer(vocabulary=vectorVocab, token_pattern=r'(?u)@?#?\b\w\w+\b')
newVectors = vec.fit_transform(df.content_cleaned)
newVectors = pd.DataFrame(newVectors.toarray(), columns=vec.get_feature_names())
df = pd.concat((df, newVectors), axis=1, join='outer', ignore_index=False, sort=False)

## send to csv and pickle
df.to_csv('twitter_database_hashtags_handles_vectors.csv', sep=',', index=None)
df.to_pickle('twitter_database_hashtags_handles_vectors_pickled.pkl')

df=pd.read_csv('twitter_database_hashtags_handles_vectors.csv', sep=',')

df_xRetweets = df[['id', 'content', 'favorites_formatted', 'content_cleaned', 'hashtags', 'mentions']][:]

#df_Set_2 = pd.DataFrame().reindex_like(df)


# *************************** multiply by retweets ****************************

# multiply new vectors by retweets, creating more new vectors
for column in df.columns[8:2262]:
    newCol = column+'_xRetweets'
    df_xRetweets[newCol] = (df['retweets_formatted']+1)*df[column]

# send to csv and pickle
df_xRetweets.to_csv('twitter_vectors_x_retweets.csv', sep=',', index=None)
df.to_csv('twitter_vectors', sep=',', index=None)

df_xRetweets.to_pickle('twitter_vectors_x_retweets.pkl')
df.to_pickle('twitter_vectors.pkl')


# *************************** resample ****************************************

# drop columns before resampling
dropCols = ['id', 'content', 'favorites_formatted', 'content_cleaned', 'hashtags', 'mentions']
df = df.drop(dropCols, axis=1)

# make new_date a datetime object before resampling and set as index
df['new_date'] = df['new_date'].apply(lambda x: datetime.strptime(x, '%m/%d/%y'))
df = df.set_index('new_date')

# resample by day
df_byDate = df.resample('D').sum()
#resample by week
df_byWeek = df.resample('W').sum()

# replace date to column
df_byDate.reset_index(inplace=True)

# replace date to column
df_byWeek.reset_index(inplace=True)


# send to csv and pickle
df_byDate.to_csv('twitter_vectors_x_retweets_by_Day.csv', sep=',', index=None)
df_byDate.to_pickle('twitter_vectors_x_retweets_by_Day_pickled.pkl')

# send to csv and pickle
df_byWeek.to_csv('twitter_vectors_x_retweets_by_Week.csv', sep=',', index=None)
df_byWeek.to_pickle('twitter_vectors_x_retweets_by_Week_pickled.pkl')

