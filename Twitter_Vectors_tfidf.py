"""

@author: Team 4
"""

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
from collections import Counter
from datetime import datetime

from nltk.corpus import stopwords
from stemming.porter2 import stem
import string
from operator import itemgetter
import spacy

from nltk.stem import WordNetLemmatizer
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer

Path='../Data/'
English_df=pd.read_csv(Path+'Twitter_Only_English_database.csv', sep=',')
English_df=English_df[['id',  'content', 'new_date', 'retweets_formatted', 'favorites_formatted']]


Path2='../Data/'

cleaned_content_df=pd.read_csv(Path2+'Content_no_hashtags_mentions_url_pic.csv', sep=',')
# *************************** definitions *************************************
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


# *************************** df cleaning *************************************

English_df = cleanDataFrame(English_df)
English_df = getHashtags(English_df)
English_df = getMentions(English_df)

# *************************** create mentions and hashtags ********************

mentionSet = getMentionSet(English_df['mentions'])
hashtagSet = getHashtagSet(English_df['hashtags'])


# create dicts from sets
mentionDict = {el:0 for el in list(mentionSet)}
hashtagDict = {el:0 for el in list(hashtagSet)}


# fill the dictionary with sum of retweets^2
for row in English_df.itertuples(index=False):
    content = row.mentions
    for item in content:
        if item in mentionDict:
            mentionDict[item] += row.retweets_formatted**2

for row in English_df.itertuples(index=False):
    content = row.hashtags
    for item in content:
        if item in hashtagDict:
            hashtagDict[item] += row.retweets_formatted**2


# drop certain mentions
mentionDict.pop('@netflix')
mentionDict.pop('@')

# drop certain hashtags
hashtagDict.pop('#netflix')
hashtagDict.pop('#netflixs')
hashtagDict.pop('#rt')
hashtagDict.pop('#')




## use the dictionaries with the squared values to:
#
#
mentionsToDrop = []
hashtagsToDrop = []

for key, value in mentionDict.items():
    if value < 111200:
        mentionsToDrop.append(key)

for key, value in hashtagDict.items():
    if value < 246230:
        hashtagsToDrop.append(key)


# drop all of the mentions and hashtags noted above
for item in mentionsToDrop:
    mentionDict.pop(item)

for item in hashtagsToDrop:
    hashtagDict.pop(item)



# create dataframes from the dictionaries
mentionDictDF = pd.DataFrame(mentionDict.items(), columns=['Mention', 'SumRetweetsSquared'])
hashtagDictDF = pd.DataFrame(hashtagDict.items(), columns=['Hashtag', 'SumRetweetsSquared'])

# csv files of the new datasets
mentionDictDF.to_csv(Path+'mentions_Final.csv', sep=',', index=None)
hashtagDictDF.to_csv(Path+'hashtags_Final.csv', sep=',', index=None)



# *************************** create vectors **********************************

# create the vocabularies for the countvectorizer
mentionVocabulary = []
for key in mentionDict.keys():
    mentionVocabulary.append(key)

hashtagVocabulary = []
for key in hashtagDict.keys():
    hashtagVocabulary.append(key)

mentionVocabulary.extend(hashtagVocabulary)
vectorVocab = mentionVocabulary

#++++++++++

# =============================================================================
# hashtags tfid
English_df['keep_hashtags']=English_df['hashtags'].apply(lambda x:  list(set(x).intersection(mentionVocabulary)))

English_df['merge_hashtags']=English_df['keep_hashtags'].apply(lambda x: " ".join(x))

tfvec_1 = TfidfVectorizer(token_pattern=r'(?u)@?#?\b\w\w+\b')
tfvec_weights_1 = tfvec_1.fit_transform(English_df['merge_hashtags']) 

df1=English_df[['new_date','favorites_formatted','retweets_formatted']][:]

# stick the weights into a new df
new_1 = pd.DataFrame(tfvec_weights_1.toarray(), columns=tfvec_1.get_feature_names())

# concat the new df with the existing df
tweet_content_vectors_1 = pd.concat((df1,new_1), axis=1, join='outer', ignore_index=False, sort=False)

# send to csv 
tweet_content_vectors_1.to_csv(Path+'twitter_hashtags_tfidf.csv', sep=',')

# =============================================================================
# mentions tfid
English_df['keep_mentions']=English_df['mentions'].apply(lambda x:  list(set(x).intersection(mentionVocabulary)))

English_df['merge_mentions']=English_df['keep_mentions'].apply(lambda x: " ".join(x))

tfvec_2 = TfidfVectorizer(token_pattern=r'(?u)@?#?\b\w\w+\b')
tfvec_weights_2 = tfvec_2.fit_transform(English_df['merge_mentions']) 

# stick the weights into a new df
new_2 = pd.DataFrame(tfvec_weights_2.toarray(), columns=tfvec_2.get_feature_names())

# concat the new df with the existing df
tweet_content_vectors_2 = pd.concat((df1,new_2), axis=1, join='outer', ignore_index=False, sort=False)

# send to csv 
tweet_content_vectors_2.to_csv(Path+'twitter_mentions_tfidf.csv', sep=',')



# =============================================================================
#content tfid




# *************************** definitions for cleaned content*********************************
def makeString(input_text):
    updated_text = str(input_text)
    return updated_text

def removePunc_1(input_text):
    input_text = str(input_text)
    regex = re.compile('[%s]' % re.escape(string.punctuation))
    updated_text =regex.sub('', input_text)
    return updated_text

def removeDigits(input_text):
    input_text = str(input_text)
    regex = re.compile('[%s]' % re.escape(string.digits)) 
    updated_text =  regex.sub('', input_text)
    return updated_text

def removeShortWords(input_text):
    input_text = str(input_text)
    updated_text = re.sub(r'\b\w{1,3}\b', '', input_text)
    return updated_text

def removeNonAscii(input_text):
    input_text = str(input_text)
    updated_text = input_text.encode("ascii", errors="ignore").decode()
    return updated_text



# ==============================for cleaned content===============================================
#string input
cleaned_content_df['tweet'] = cleaned_content_df['content_3'].apply(makeString)

# make lowercase
cleaned_content_df['tweet_lower'] = cleaned_content_df['tweet'].apply(lambda x: " ".join(x.lower() for x in x.split()))
    
# remove punctuation
cleaned_content_df['tweet_punct'] = cleaned_content_df['tweet_lower'].apply(removePunc_1)

# remove digits
cleaned_content_df['tweet_dig'] = cleaned_content_df['tweet_punct'].apply(removeDigits)

#remove no ascii
cleaned_content_df['tweet_no_ascii']=cleaned_content_df['tweet_dig'].apply(removeNonAscii)

#remove short

cleaned_content_df['tweet_short']=cleaned_content_df['tweet_no_ascii'].apply(removeShortWords)


# remove stop words
stop = stopwords.words('english')
newStop = ['netflixs','netflix', 'nflx']
stop.extend(newStop)
cleaned_content_df['tweet_stop'] = cleaned_content_df['tweet_short'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))

# lemmatize
nlp = spacy.load('en', disable=['parser', 'ner'])
cleaned_content_df['tweet_lemma'] = cleaned_content_df['tweet_stop'].apply(lambda x: " ".join([token.lemma_ for token in nlp(x)]))
# fit a TFIDF vectorizer, manipulate the parameters below as needed
# for this analysis, I've run only 2 grams (not 1 or >2) and obtained max of 100 word features/columns
# more features would be better for algorithmic reasons - I used 100 as a start only
tfvec = TfidfVectorizer(min_df=.001, max_df=.9, ngram_range=(2,3))
tfvec_weights = tfvec.fit_transform(cleaned_content_df['tweet_lemma']) 

# stick the weights into a new df
new = pd.DataFrame(tfvec_weights.toarray(), columns=tfvec.get_feature_names())

# concat the new df with the existing df
tweet_content_vectors_0 = pd.concat((df1,new), axis=1, join='outer', ignore_index=False, sort=False)

# send to csv 
tweet_content_vectors_0.to_csv(Path+'twitter_words_tfidf.csv', sep=',')


# =============================================================================


# concat the new df with the existing df
tweet_content_vectors = pd.concat((df1,new_1), axis=1, join='outer', ignore_index=False, sort=False)
tweet_content_vectors = pd.concat((tweet_content_vectors,new_2), axis=1, join='outer', ignore_index=False, sort=False)
tweet_content_vectors = pd.concat((tweet_content_vectors,new), axis=1, join='outer', ignore_index=False, sort=False)

# make new_date a datetime object before resampling and set as index
tweet_content_vectors['new_date'] = tweet_content_vectors['new_date'].apply(lambda x: datetime.strptime(x, '%m/%d/%y'))
tweet_content_vectors = tweet_content_vectors.set_index('new_date')

tweet_content_vectors.to_csv(Path+'tweet_tfidf_vectors.csv', sep=',')

# =============================================================================
# build vectors by weeks
# #check for zeros L2 norm
L2_token={}
for column in list(set(tweet_content_vectors.columns)-{'new_date', 'favorites_formatted', 'retweets_formatted'}):
    L2_token[column]=np.sqrt(sum(np.square(tweet_content_vectors[column])))

L2_tokenDF = pd.DataFrame(L2_token.items(), columns=['Feature', 'L2'])
pfr = pandas_profiling.ProfileReport(L2_tokenDF)
pfr.to_file(Path+'Quick_Stats_summary_L2_token.html')

# =============================================================================
# build vectors by weeks x retweets
# multiply new vectors by retweets, creating more new vectors

tweet_content_vectors_xRetweets=tweet_content_vectors[['favorites_formatted', 'retweets_formatted']]
#initial_vectors_weigthed=pd.read_csv(Path+'twitter_vectors_x_retweet_likes_byWeek.csv', parse_dates=True, index_col=0)
#initial_vectors_weigthed.rename(index=str, columns={"target": "TARGET"}, inplace=True)

for column in tweet_content_vectors.columns[2:tweet_content_vectors.shape[1]]:
#    newCol = column+'_xRetweets'
    tweet_content_vectors_xRetweets[column] = (tweet_content_vectors['retweets_formatted']+1)*tweet_content_vectors[column]

dropCols=['favorites_formatted', 'retweets_formatted']
tweet_content_vectors_xRetweets.drop(dropCols, axis=1,inplace=True)

# build vectors by weeks
tweet_vectors_xRetweets_byWeek = tweet_content_vectors_xRetweets.resample('W').sum()


tweet_vectors_xRetweets_byWeek=pd.read_csv(Path+'Twitter_vectors_byWeek_tfidf.csv', parse_dates=True, index_col=0)


Path2='../Data/'

#bring sentiment vectors
df=pd.read_csv(Path2 +'Weighted_tweets_byWeek_2_3grams_mentions_hashtags_Xretweets.csv', parse_dates=True, index_col=0)

#attach sentiment
tweet_vectors_xRetweets_byWeek[['Positive', 'Negative'] ]=df[['POSITIVE', 'NEGATIVE']]

tweet_vectors_xRetweets_byWeek.to_csv(Path+'Twitter_vectors_byWeek_tfidf.csv', sep=',')

 

