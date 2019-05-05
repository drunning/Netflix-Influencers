#!/usr/bin/env python
# coding: utf-8

"""
@author: Team 4

"""

import pandas as pd
from os import listdir

PATH = 'D:/data/raw/twitter/'
FILE_OUT = 'D:/data/processed/twitter/twitter_data.csv'
ERROR_OUT = 'D:/data/logs/twitter_encoding_errors.csv'


files = listdir(PATH)
twitter_list = []
error_list = []
files_read=0
keep_columns = ['handle', 'name', 'content', 'replies', 'retweets', 'favorites', 'unix_timestamp', 'published_date', 'url']


for file in files:
    if file.endswith('.csv'):
        files_read+=1
        try:
            df = pd.read_csv(PATH+file, index_col=None, header=0)
            df = df[keep_columns]
            twitter_list.append(df)
        except:
            error_list.append(file)


twitter_df = pd.concat(twitter_list, axis=0, ignore_index=True)
twitter_df.to_csv(FILE_OUT, sep=',')


if len(error_list)>0: 
    error_df = pd.DataFrame(error_list)
    error_df.columns = ['file']
    error_df.to_csv(ERROR_OUT, sep=',')


print("Files read: ", files_read)
print("Files with no errors: ", len(twitter_list))
print("Files with encoding errors: ", len(error_list))
