#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Team 4
"""

import datetime as dt
import matplotlib.pyplot as plt
from matplotlib import style
import pandas as pd
import pandas_datareader.data as web
from ta import *

Path='/Users/crisdarley/Desktop/Vectors/Data/'
Data_Path='/Users/crisdarley/Desktop/ML/Data/'
Pic_Path='/Users/crisdarley/Desktop/ML/Pic/'
start=dt.datetime(2016,12,1)
end=dt.datetime(2019,3,17)

Ticker_list=['NFLX']#, '^GSPC','^DJI', '^IXIC']
Names_list= ['Netflix']#, 'SP500','Dow', 'Nasdaq' ]

for i in range(1):
    stock=web.DataReader(Ticker_list[i],'yahoo',start,end)
    stock['Ave_Volume']=stock['Volume'][:]
    logic = {'Open'  : 'first',
             'High'  : 'max',
             'Low'   : 'min',
             'Close' : 'last',
             'Adj Close': 'last',
             'Volume': 'sum',
             'Ave_Volume':'mean'}
#offset = pd.offsets.timedelta(days=-7)
#f=stock.resample('W', loffset=offset).apply(logic)
    w=stock.resample('W').apply(logic)
    m=stock.resample('M').apply(logic)
#% change close_price    
    stock['target']=stock.Close.pct_change()
    w['target']=w.Close.pct_change()
    m['target']=m.Close.pct_change()
#duplicate % change close price    
    stock['T1_price_change']=stock.Close.pct_change()
    w['T1_price_change']=w.Close.pct_change()
    m['T1_price_change']=m.Close.pct_change()
#% change Total volumne 
    stock['T2_Vol_change']=stock.Volume.pct_change()
    w['T2_Vol_change']=w.Volume.pct_change()
    m['T2_Vol_change']=m.Volume.pct_change()
#% change Total volumne 
    stock['T2_AveVol_change']=stock.Ave_Volume.pct_change()
    w['T2_AveVol_change']=w.Ave_Volume.pct_change()
    m['T2_AveVol_change']=m.Ave_Volume.pct_change()

    File_out_daily= Names_list[i]+ '_daily.csv'
    File_out_weekly=Names_list[i]+ '_weekly.csv'
    File_out_monthly=Names_list[i]+ '_montly.csv'
    
    
    stock.to_csv(File_out_daily)
    w.to_csv(Data_Path+File_out_weekly)
    m.to_csv(Data_Path+File_out_monthly)  
#w.to_csv(Data_Path+'price_stock_weekly.csv', sep=',')

