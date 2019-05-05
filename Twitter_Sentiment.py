import spacy
from textblob import TextBlob
from textblob import Blobber
from textblob.sentiments import NaiveBayesAnalyzer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import re
import pandas as pd
import numpy as np

# Modify the paths as needed.  These are from my os/filing system
FILE = '../Data/twitter_base_data_v2.csv'
FILE_OUT = '../Data/twitter_sentiments_2.csv'

# read into df
df = pd.read_csv(FILE)


# ************************* CLEANING DEFINITIONS **************************************
def removeShortWords(input_text):
    input_text = str(input_text)
    updated_text = re.sub(r'\b\w{1,3}\b', '', input_text)
    return updated_text

# df is a required parameter.  All other are optional to control what preprocessing to do
def cleanText(df, removeHash=True, removeURL=True, removeImg=True, \
removeHandle=True, makeLower=True, removePunc=True, \
removeStop=False, removeShort=False, getLemma=False):

    if removeHash:
        df['textForSentiment'] = df['content'].apply(lambda x: " ".join([word for word in x.split() if not word.startswith('#')]))

    if removeURL:
        urlPattern = re.compile('((www\.[^\s]+)|(https?://[^\s]+))')
        df['textForSentiment'] = df['textForSentiment'].apply(lambda x: " ".join([word for word in x.split() if urlPattern.match(word) is None]))

    if removeImg:
        imageFormats = ['.jpg', '.png', '.gif', '.tif']
        df['textForSentiment'] = df['textForSentiment'].apply(lambda x: " ".join([word for word in x.split() if not word[-4:] in imageFormats]))

    if removeHandle:
        df['textForSentiment'] = df['textForSentiment'].apply(lambda x: re.sub(r'@[A-Za-z0-9]+','', x))

    if makeLower:
        df['textForSentiment'] = df['textForSentiment'].apply(lambda x: " ".join([word.lower() for word in x.split()]))

    if removePunc:
        df['textForSentiment'] = df['textForSentiment'].str.replace('[^\w\s]', '')

    if removeStop:
        stop = stopwords.words('english')
        df['textForSentiment'] = df['textForSentiment'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))

    if removeShort:
        df['textForSentiment'] = df['textForSentiment'].apply(removeShortWords)

    if getLemma:
        nlp = spacy.load('en', disable=['parser', 'ner'])
        df['textForSentiment'] = df['textForSentiment'].apply(lambda x: " ".join([token.lemma_ for token in nlp(x)]))

    return df

# ************************** SENTIMENT SCORE DEFINITIONS ********************************
def textblobSentiment(sentence):
    score = TextBlob(sentence).sentiment
    return score.polarity


def textblobSubjectivity(sentence):
    score = TextBlob(sentence).sentiment
    return score.subjectivity


tb = Blobber(analyzer=NaiveBayesAnalyzer())
def textblobSentimentNB(sentence):
    blob = tb(sentence)
    return blob.sentiment.classification


analyser = SentimentIntensityAnalyzer()
def sentiment_analyzer_scores(sentence):
    score = analyser.polarity_scores(sentence)
    score = score['compound']
    return score

# ************************* GET SENTIMENT SCORES **************************************

# call the cleanText function.  Pass parameters depending on what cleaning
# methods to run.  See lines 25-27.  If you do not pass any parameters, the
# defaults will be used (as indicated by the = in lines 25-27)
df = cleanText(df)


# get sentiment scores and put in new columns
df['textblobPolarity'] = df['textForSentiment'].apply(textblobSentiment)
df['textblobSubjectivity'] = df['textForSentiment'].apply(textblobSubjectivity)
df['textblobPolarity_NB'] = df['textForSentiment'].apply(textblobSentimentNB)
df['vaderScore'] = df['textForSentiment'].apply(sentiment_analyzer_scores)


# send to csv
df.to_csv(FILE_OUT, sep=',', index=None)
