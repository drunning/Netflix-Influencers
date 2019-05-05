#!/usr/bin/env python
# coding: utf-8
"""

@author: Team 4

****
The script does:
    1. read the aggregated cvs file
    2. remove tweets older than '12/31/16' and  newer than '2/28/19'
    3. remove duplicate tweets 
    4. remove any columns added by the web scraper app 
        left with: 'content', 'favorites', 'handle', 'name','unix_timestamp',
        'published_date', 'replies','retweets','url'
    4. fill with zeros the empty values on the columns favorites','replies', 'retweets'
    5. order by published_date
    6. split published_date into two columns 'new_date', 'time'
    7. create dictionary for "url" -> {'url':{<id>: <actual_link>}}
    8. create dictionary for actual tweets  -> {<id>:{'content': ***, ....,'retweet':**}}
       this dictionary includes 'time' and 'new_date' but it doesn't include the 'url'
    9. make a list of dictionaries and dataframes
    10. save the list of dictionaries and dataframes to 'twitter_base_data.pickle'

"""
import pickle
import numpy as np
import pandas as pd
from langdetect import detect


FILE_IN ='D:/data/processed/twitter/twitter_data.csv'


twitter_df = pd.read_csv(FILE_IN)

##to remove duplicates tweets--> unique 'url'
twitter_df.drop_duplicates(subset=['url'], keep='first', inplace=True)
##
twitter_df.reset_index(drop=True, inplace=True)


##wanted columns, removing: columns added by web scraping app
##
wanted=set([ 'handle','name','content','replies','retweets','favorites',
            'unix_timestamp','published_date','url'])
unwanted=set(list(twitter_df.columns))

drop_col=list(unwanted-wanted)
twitter_df.drop(drop_col, axis=1, inplace=True)

##
twitter_df.dropna(axis=0, subset=['published_date'],inplace=True)

twitter_df.loc[twitter_df["name"].isnull(),'name'] = twitter_df["handle"] 
##
##fill with zeros NaN for 'favorites' 'replies' 'retweets'

twitter_df.fillna(0, inplace=True)
twitter_df.reset_index(drop=True,inplace=True)

##splits published_date to new_date and hour
publisheddate=twitter_df['published_date'].str.split(" ", n = 1, expand = True)
twitter_df['time']=publisheddate[1]
twitter_df['new_date']=publisheddate[0]

#twitter_df.sort_values(by='new_date') 

#remove  tweets older than '12/31/16' and  newer than '2/28/19'
#base_df
###upper_bound ('3/1/19')
#ub=twitter_df[twitter_df['published_date'].ge('3/1/19')]
#
####lower_bound ('12/31/16')
#lb=twitter_df[twitter_df['published_date'].le('12/31/16')]
#
#indices=list(lb.index)+ list(ub.index)
#
#twitter_df.drop(indices, inplace=True)
#
#twitter_df.reset_index(drop=True,inplace=True)

###sort tweets by publisheddate: 0 is the newest
#twitter_df['published_date'] =pd.to_datetime(twitter_df.published_date)
#twitter_df.sort_values(by='published_date') 
#twitter_df.reset_index(drop=True, inplace=True)


#### subset data frames
## creates data frame for 'ids' and 'url' unique values
Id_df=twitter_df[['url']].copy(deep=True)
##
###data frame for tweets
#remove url  since id is the url value

tweets_df=twitter_df[[ 'handle','name','content','replies','retweets',
                      'favorites','unix_timestamp','published_date',
                      'time','new_date']].copy(deep=True)

### create dictionaries for id and for tweets
Id_dict=Id_df.to_dict()
tweets_dict=tweets_df.to_dict('index')
##
###list of data frames and dictionaries: ids and tweets
l=[Id_df,Id_dict, tweets_df, tweets_dict]


##saving with pickle module to a file
FILE_OUT = 'D:/data/processed/twitter/twitter_base_data.pickle'


pickle_out=open (FILE_OUT, 'wb')
pickle.dump(l,pickle_out )
pickle_out.close()

##saving csv

FILE_OUT = '../Data/twitter_base_data.csv'
twitter_df.to_csv(FILE_OUT, sep=',')


FILE = '../Data/twitter_base_data.csv'
FILE_OUT = '../Data/twitter_nonEnglish.csv'

df = pd.read_csv(FILE)


def detectLanguage(content):
    exceptions = []
    try:
        language = detect(content)
        return language
    except:
        exceptions.append(content)
        return "NA"


df['language'] = df['content_'].apply(detectLanguage)
df.to_csv(FILE_OUT, sep=',', index=False)

English_only_df=df[df['nationality'] == "en"]

FILE_OUT = '../Data/Twitter_Only_English_database.csv'
English_only_df.to_csv(FILE_OUT, sep=',', index=False)

