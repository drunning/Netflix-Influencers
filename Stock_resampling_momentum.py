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

start=dt.datetime(2017,1,1)
end=dt.datetime(2019,2,28)


start=dt.datetime(2016,9,1)
end=dt.datetime(2019,2,28)
our_start_date=dt.datetime(2017,1,1)

nflx=web.DataReader('NFLX','yahoo',start,end)


for i in [5,10,14,20]:
## =============================================================================
##https://www.investopedia.com/terms/r/rsi.asp 
##ta.momentum.rsi(close, n=14, fillna=False)
## =============================================================================
#
    nflx_osc=nflx[:]
    nflx_osc['RSI'] = rsi(nflx_osc["Close"], n=i,fillna=False)

## =============================================================================
## https://www.investopedia.com/terms/s/stochasticoscillator.asp
##ta.momentum.stoch(high, low, close, n=14, fillna=False) 
## =============================================================================


    nflx_osc['Stoch_osc'] = stoch(nflx_osc['High'],nflx_osc['Low'],nflx_osc["Close"], n=i,fillna=False)

# =============================================================================
# #Drop extra values
# =============================================================================
    #print(nflx_osc.head(3))
    nflx_osc.drop(nflx_osc[nflx_osc.index <= our_start_date].index, inplace=True)
    #print(nflx_osc.head(3))

    nflx_osc['RSI_cond']=nflx_osc.apply( fRSI, axis=1)
    nflx_osc['Stoch_cond']=nflx_osc.apply( fStoch, axis=1)
    nflx_osc['Over']=nflx_osc.apply( fOver, axis=1)

    nflx_osc.drop(nflx_osc[nflx_osc.index <= our_start_date].index, inplace=True)

    FILE_OUT='Momentum_'+ str(i) + '.csv'
#

    nflx_osc.to_csv(FILE_OUT)
    
for i in [5,10,14,20]:
    Root='Momentum_'+ str(i)
    FILE_IN=Root + '.csv'
    df=pd.read_csv(FILE_IN,parse_dates=True, index_col=0)

#Ticker_list=['NFLX', '^GSPC','^DJI', '^IXIC']
#Names_list= ['Netflix', 'SP500','Dow', 'Nasdaq' ]
#
#for i in range(4):
#    stock=web.DataReader(Ticker_list[i],'yahoo',start,end)
#    stock['Ave. Volume']=stock['Volume']
    logic = {'Open'  : 'first',
             'High'  : 'max',
             'Low'   : 'min',
             'Close' : 'last',
             'Adj Close': 'last',
             'Volume': 'sum',
             'RSI':'mean',
             'Stoch_osc':'mean'}
    d=df.resample('W').apply(logic)
    
    d['RSI_cond']=d.apply( fRSI, axis=1)
    d['Stoch_cond']=d.apply( fStoch, axis=1)
    d['Over']=d.apply( fOver, axis=1)
    
    m=df.resample('M').apply(logic)
    m['RSI_cond']=m.apply( fRSI, axis=1)
    m['Stoch_cond']=m.apply( fStoch, axis=1)
    m['Over']=m.apply( fOver, axis=1)    
    
    
    File_out_daily= Root + '_daily.csv'
    File_out_weekly=Root+ '_weekly.csv'
    File_out_monthly=Root+ '_montly.csv'
    
    df.to_csv(File_out_daily)
    d.to_csv(File_out_weekly)
    m.to_csv(File_out_monthly)  



