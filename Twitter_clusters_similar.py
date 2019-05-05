#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 16:06:00 2019

@author: crisdarley
"""

import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from operator import itemgetter

import math
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.cluster import DBSCAN
import statsmodels.api as sm # import statsmodels
from sklearn.decomposition import PCA
import operator
from difflib import SequenceMatcher

Path1='../clusters/'
Path='../Data/'
Data_Path='../Data/'
Pic_Path='../Pic/'


path='../Data/similar/'



#this are vectors weighted but I removed the '_xRetweets' ftr=feature
df_ftr=pd.read_csv(Path +'Twitter_vectors_byWeek_tfidf.csv', parse_dates=True, index_col=0)

#this targets==>tgt
df_tgt=pd.read_csv(Data_Path+'Targets_vectors_byWeek.csv', parse_dates=True, index_col=0)
#Stats=df.describe()


#Clusters
df_clusters=pd.read_csv(Path+'Twitter_clusters.csv', parse_dates=True, index_col=0)
df_clusters.reset_index(level=0, inplace=True)
df_clusters.rename(columns={'index':'features'},inplace=True)


df_ftr=pd.read_csv(Path +'Twitter_vectors_byWeek_tfidf.csv', parse_dates=True, index_col=0)

df_ftr.drop(columns=['Positive', 'Negative'],inplace=True)
# =============================================================================
def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()
# =============================================================================
def getSimilar(c):
    A=c[:]
    B=c[:]
    sim={}
    while len(B)>0:
        i=0
        while i <len(A):
            j=0
            while j < len(B):
                if similar(A[i],B[j]) > .5:
                    if A[i] in sim:
                        sim[A[i]].append(B[j])
                        B.remove(B[j])
                        
                    else:
                        sim[A[i]]=[B[j]]
                        B.remove(B[j])
                j=j+1
            i=i+1
            A=B[:]
    return sim
# =============================================================================


# make a dictionary of cluster methods and corresponding cluster labels and members
clusterMaster = {}
for clusterName in df_clusters.columns[1:]:
    clusters = {}
    clusterMaster[clusterName] = clusters
    for feature, cluster in zip(df_clusters['features'].values ,df_clusters[clusterName].values):
        if cluster in clusters:
            clusters[cluster].append(feature)
        else:
            clusters[cluster] = [feature]

sel_cluster_df=pd.read_csv(Data_Path +'Selected_clusters_sorted.csv', parse_dates=False)
sel_cluster_df['similar']=''
sel_cluster_df['new_size']=''

for i in range(len(sel_cluster_df)):
    alg_name=sel_cluster_df.name.loc[i]
    
    members = clusterMaster[sel_cluster_df.algorithm.loc[i]][sel_cluster_df.cluster_num.loc[i]]
    sim_ftr=getSimilar(members)
    df = pd.DataFrame([sim_ftr], columns=sim_ftr.keys())
    df1= df.T.reset_index()
    df1.rename(columns={'index':'new_member',0:'similar_members'}, inplace=True)
    vectors=pd.DataFrame(index=df_ftr.index)
    for j in range(len(df1)):
        vectors[df1.new_member.loc[j]]=df_ftr[df1.similar_members.loc[j]].sum(axis=1)
    file_out='new_vectors_'+ alg_name+'.csv'
    vectors.to_csv(path + file_out,sep=',')
    file_out_2='similar_features_'+ alg_name+'.csv'
    df1.to_csv(path + file_out_2,sep=',')
    A=list(zip(df1.new_member,df1.similar_members))
    sel_cluster_df['similar'].loc[i]=A
    sel_cluster_df['new_size'].loc[i]=len(A)
    
sel_cluster_df.to_csv(Path1 +'Selected_clusters_modify.csv', sep=',')    


