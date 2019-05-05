#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: Team 4
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


Path='../Data/'
Data_Path='../Data/'
Pic_Path='../Pic/'



#this are vectors weighted but I removed the '_xRetweets' ftr=feature
df_ftr=pd.read_csv(Path +'Twitter_vectors_byWeek_tfidf.csv', parse_dates=True, index_col=0)

df_ftr.drop(columns=['Positive', 'Negative'],inplace=True)
#this targets==>tgt
df_tgt=pd.read_csv(Path +'Targets_vectors_byWeek.csv', parse_dates=True, index_col=0)
#Stats=df.describe()

#standardize the data with respect to normal distribution and save them to dataframe

df_ftr_scaled = preprocessing.scale(df_ftr)
df_ftr_std = pd.DataFrame(data=df_ftr_scaled, index=df_ftr.index, columns=df_ftr.columns)

df_tgt_scaled = preprocessing.scale(df_tgt)
df_tgt_std = pd.DataFrame(data=df_tgt_scaled, index=df_tgt.index, columns=df_tgt.columns)

#Some stats
Stats_ftr=df_ftr_std.describe()
Stats_tgt=df_tgt_std.describe()

 #create empty cluster with index from columns of features
df_clusters= pd.DataFrame(index=df_ftr.T.index)

# =============================================================================
#Generate several K-means clusters (size depends on analysis from K-means cluster)
# =============================================================================
for j in [80,91, 102, 114]:
# Fitting K-Means to the dataset
    kmeans = KMeans(n_clusters = j, init = 'k-means++', random_state = 100)
    y_kmeans = kmeans.fit_predict(df_ftr_std.T)    
# name of clusters    
    new_clusters='cluster_Kmeans_'+str(j)
#    df_clusters= pd.DataFrame(y_kmeans)
#    new_clusters='cluster_'+str(j)
# Adding cluster to the Dataset1
    df_clusters[new_clusters] = y_kmeans

# =============================================================================
# # AgglomerativeClustering clustering  euclidean -minimize the variant 
#between clusters
# =============================================================================
from sklearn.cluster import AgglomerativeClustering

n_clusters=80

clusters = AgglomerativeClustering(n_clusters=80, affinity='euclidean', linkage='ward')  
y_Agglomerative=clusters.fit_predict(df_ftr_std.T) 

df_clusters['Agglomerative_80'] = y_Agglomerative

# =============================================================================
# # AgglomerativeClustering clustering  euclidean -minimize the variant 
#between clusters
# =============================================================================

clusters = AgglomerativeClustering(n_clusters=n_clusters, linkage='average', 
                                     affinity='cosine')
#average uses the average of the distances of each observation of the two sets.
y_Agglomerative=clusters.fit_predict(df_ftr_std.T) 
df_clusters['Agglomerative_80_cos'] = y_Agglomerative

# =============================================================================
# # AgglomerativeClustering clustering  euclidean -minimize the variant 
#between clusters
# =============================================================================
from sklearn.metrics.pairwise import cosine_distances
#precompute distance
dist_cos = cosine_distances(df_ftr_std.T)

clusters = AgglomerativeClustering(n_clusters, affinity='precomputed', linkage='average')

y_Agglomerative=clusters.fit_predict(dist_cos)
df_clusters['Agglomerative_80_cos_precom']=y_Agglomerative
# =============================================================================
# hierarchical clustering-flat
# =============================================================================
# creating a dataset for hierarchical clustering
# Assigning the clusters and plotting the observations as per hierarchical clustering

k=40
np.set_printoptions(precision=40, suppress=True)  # suppress scientific float notation
#creating the linkage matrix
H_cluster = linkage(df_ftr_scaled.T,'ward')
#plot
#dendrogram(
#    H_cluster,
#    truncate_mode='lastp',  # show only the last p merged clusters
#    p=40,  # show only the last p merged clusters
#    leaf_rotation=90.,
#    leaf_font_size=12.,
#    show_contracted=True,  # to get a distribution impression in truncated branches
#)
#plt.show()

clusters = fcluster(H_cluster, k, criterion='maxclust')

df_clusters['Hierarchical_ward']=clusters 

#creating the linkage matrix
H_cluster = linkage(dist_cos,'weighted')



cluster = fcluster(H_cluster, k, criterion='maxclust')

df_clusters['H_flat_weighted']=cluster 

# =============================================================================
# DBSCAN
# =============================================================================


# create clusters
clusters = DBSCAN(eps=0.1, min_samples=5, metric='cosine')
#min epsilon=.01
y_DBSCAN=clusters.fit_predict(df_ftr_scaled.T)
df_clusters['DBSCAN_01_Cos']=y_DBSCAN

# create clusters
clusters = DBSCAN(eps=0.1, min_samples=5, metric='precomputed')
#min epsilon=.01
y_DBSCAN=clusters.fit_predict(dist_cos)
df_clusters['DBSCAN_01_cos_sim']=y_DBSCAN

# create clusters
clusters = DBSCAN(eps=0.1, min_samples=5, metric='euclidean')
#min epsilon=.01
y_DBSCAN=clusters.fit_predict(df_ftr_scaled.T)
df_clusters['DBSCAN_01_Euc']=y_DBSCAN


clusters = DBSCAN(eps=0.1, min_samples=5, metric='minkowski', p=3)
#min epsilon=.01
y_DBSCAN=clusters.fit_predict(df_ftr_scaled.T)
df_clusters['DBSCAN_01_Mink']=y_DBSCAN


df_clusters.to_csv(Path+'Twitter_clusters.csv', sep=',')


#Clusters
df_clusters=pd.read_csv(Path+'Twitter_clusters.csv', parse_dates=True, index_col=0)
df_clusters.reset_index(level=0, inplace=True)
df_clusters.rename(columns={'index':'features'},inplace=True)

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


###regression
def getRegression1(clusters):
    R2={}
    for key,value in clusters.items() :
        new_key=str(key)

        X=df_ftr[list(value)]
        X = sm.add_constant(X)
        X_scaled = preprocessing.scale(X)

        Y=df_tgt['price_change'].T
        Y_scaled = preprocessing.scale(Y)
        N1=math.floor(X.shape[1]*.8)
        N2=min(N1,80)
        N=max(1,N2)
        pca = PCA(n_components=N)
        X_pca = pca.fit_transform(X_scaled)

    #minimizing least squares
        model = sm.OLS(Y_scaled, X_pca).fit()
        model.summary()
        R2[new_key]=round(model.rsquared_adj,4)

    #R2_sig = dict((k, v) for k, v in R2.items() if v >= .5)
    R2_sig = [(k, v) for k, v in R2.items() if v >= .5 ]
    R2_sig = sorted(R2_sig, key=itemgetter(1),reverse=True)
    return R2_sig


###regression 2
def getRegression2(clusters):
    R2_shift={}
    for key,value in clusters.items() :
        new_key=str(key)

        X=df_ftr[list(value)]
        X = sm.add_constant(X)
        X_scaled = preprocessing.scale(X)

        Y=df_tgt['price_change_shift'].T
        Y_scaled = preprocessing.scale(Y)
        N1=math.floor(X.shape[1]*.8)
        N2=min(N1,80)
        N=max(1,N2)
        pca = PCA(n_components=N)
        X_pca = pca.fit_transform(X_scaled)
    #minimizing least squares
        model = sm.OLS(Y_scaled, X_pca).fit()
        model.summary()
        R2_shift[new_key]=round(model.rsquared_adj,4)


# columns for new dataframe to be created
cols = ['algorithm', 'cluster_num','R2','size_members', 'R2_members']    
#cols = ['cluster', 'R2_sig', 'R2_shift_sig', 'R2_vol_sig', 'R2_vol_shift_sig', 'R2_max', 'R2_max_members']
rows = []

# create the rows for the dataframe
for col in clusterMaster.keys():
    clusters={}
    for feature, cluster in zip(df_clusters['features'].values ,df_clusters[col].values):
        if cluster ==-1:
            continue
        else:
            if cluster in clusters:
                clusters[cluster].append(feature)
            else:
                    clusters[cluster] = [feature]
    R2_sig = getRegression1(clusters)
    for i in range(len(R2_sig)):
        clusterNumber=int(R2_sig[i][0])
        members = clusterMaster[col][clusterNumber]
        r2=R2_sig[i][1]
        mem_len=len(members)
        rows.append([col,clusterNumber, r2  ,mem_len,members])


clusterSummaryDF = pd.DataFrame(rows, columns=cols, index=None)
clusterSummaryDF.to_csv(Path+'clusterR2Summary.csv', index=None)  
    
    
    
cols = ['algorithm', 'cluster_num','R2','size_members', 'R2_members']    
#cols = ['cluster', 'R2_sig', 'R2_shift_sig', 'R2_vol_sig', 'R2_vol_shift_sig', 'R2_max', 'R2_max_members']
rows = []

# create the rows for the dataframe
for col in clusterMaster.keys():
    clusters={}
    for feature, cluster in zip(df_clusters['features'].values ,df_clusters[col].values):
        if cluster ==-1:
            continue
        else:
            if cluster in clusters:
                clusters[cluster].append(feature)
            else:
                    clusters[cluster] = [feature]
    R2_sig = getRegression2(clusters)
    for i in range(len(R2_sig)):
        clusterNumber=int(R2_sig[i][0])
        members = clusterMaster[col][clusterNumber]
        r2=R2_sig[i][1]
        mem_len=len(members)
        rows.append([col,clusterNumber, r2  ,mem_len,members])


clusterSummaryDF_shift = pd.DataFrame(rows, columns=cols, index=None)
clusterSummaryDF_shift.to_csv(Path+'clusterR2_shift_Summary.csv', index=None)      
    


