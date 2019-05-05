#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  3 13:13:51 2019

Creates csv file  with delimited by "|"
"""

# calling bs4 and using beautifulSoup
import bs4
from urllib.request import urlopen as uReq
# alias to beautifulSoup
from bs4 import BeautifulSoup as soup    

#from datetime import datetime


#create cvs file
filename = "Netflix_News.csv"
f=open(filename,"w")

headers="date | type | title | webpage | directLink \n"

f.write(headers)


year=[2017,2018,2019]

for y in year:
#page
    my_url='https://www.netflixinvestor.com/investor-news-and-events/financial-releases/' + str(y) + '/default.aspx'
#    print(my_url)
#open webconnection 
    uClient = uReq(my_url)

    page_html=uClient.read()

    uClient.close()

# html parsing 
    page_soup = soup(page_html, "html.parser")

#grab news
    urlsContainers=page_soup.find_all("div", {"class":"module_headline"})
    datesContainers=page_soup.find_all("div", {"class":"module_date-time"})
    linkContainers=page_soup.find_all("div",{"class":"module_links"})
    for i in range(len(urlsContainers)):

        title =urlsContainers[i].a.text.strip()
        if ("Releases" in title):
            newsType="results release"
        elif  ("Announce" in title):
            newsType="results anouncement"
        else:
            newsType="other announcement"
        url=urlsContainers[i].a.get("href")
        date=datesContainers[i].text.strip()        
        link1=linkContainers[i].a.get("href")
        link2=link1[2:]
    
        f.write(date + " | " + newsType + " | " + title.replace(","," -&- ") + " | " + url + " | " + link2 + "\n")

f.close()
