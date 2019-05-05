# -*- coding: utf-8 -*-
"""

@author: Team 4

"""
import numpy as np
import pandas as pd
from sklearn.metrics import *
from sklearn import metrics

## Path needs to be added if it files aren't in the current folder
FILE_IN='Scored_random_set.csv'
df_actual=pd.read_csv(FILE_IN, sep=',')
df_actual.head()

def cat(score,threshold):
    if score <-threshold:
        return -1
    if score > threshold:
        return 1
    return 0

def metrics_report_to_df(ytrue, ypred):
    precision, recall, fscore, support = metrics.precision_recall_fscore_support(ytrue, ypred)
    classification_report = pd.concat(map(pd.DataFrame, [precision, recall, fscore, support]), axis=1)
    classification_report.columns = ["precision", "recall", "f1-score", "support"] # Add row w "avg/total"
    classification_report.loc['avg/Total', :] = metrics.precision_recall_fscore_support(ytrue, ypred, average='weighted')
    classification_report.loc['avg/Total', 'support'] = classification_report['support'].sum() 
    return(classification_report)

summary_df=pd.DataFrame(columns=['File','threshold','summary_score','accuracy', 'precision', 'recall', 'f1-score', 'support'])
empty_df=pd.DataFrame({'File':['.'],'threshold':['.'],'summary_score':['.'],'accuracy':['.'], 'precision':['.'], 'recall':['.'], 'f1-score':['.'], 'support':['.']} )
for member in ['Jason','Jesse']:
    for i in [1,2,3]:
        FILE_IN_1 ='sentiment_'+member+'_'+ str(i) + '.csv'
    #FILE = 'D:/data/processed/twitter/twitter_base_data.csv'
        print(FILE_IN_1)
        df_scored= pd.read_csv(FILE_IN_1, sep=',')
        df_scored.head()
        df_results=pd.merge(df_actual, df_scored, on='id', how='inner')
    
    #for col in df.columns:
        for col in ['textblobPolarity','textblobSubjectivity', 'vaderScore']:
                for threshold in [0.34, 0.5, 0.75]:
                    column_name= col +'_'+ str(threshold)
                    df_results[column_name]=df_results[col].apply(lambda x: cat(x,threshold))
                    class_report = metrics_report_to_df(df_results['Sentiment'], df_results[column_name])
                    class_report['accuracy']= accuracy_score(df_results['Sentiment'], df_results[column_name])
                    class_report['File']=FILE_IN_1
                    class_report['threshold']=threshold
                    class_report['summary_score']=column_name[:len(col)]
                    summary_df=summary_df.append(class_report,ignore_index=False, sort=False)
                    summary_df=summary_df.append(empty_df,ignore_index=False, sort=False)
                    column_name=column_name[:len(col)]                   
    summary_df.to_csv(member +'_summary_metrics.csv')
    df_results.to_csv(member+'_classification.csv')
    
#
