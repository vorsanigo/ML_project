from simple_image_download import simple_image_download as simp
import os 


main_path =  'C:/Users/alber/Desktop/UniTn/Data Science/Second_Semester/Machine_Learning/Competition/Img_scarp/simple_images'
immagini = ['sagrada familia', 'colosseum', 'empire state building, eiffel tower', 'arena di verona', 'pantheon', 'taj mahal']

for name in immagini :


    response = simp.simple_image_download
    response().download(name, 5)

    print(response().urls(name, 5))














'''#from typing_extensions import Required
from urllib import request
from bs4.element import PageElement
from requests import Request, Session
from bs4 import BeautifulSoup
import os 
import requests
import urllib
from urllib import parse 
import base64
import re


#Firstly, set search_words which you want to have the image of
search_words = ['Colosseum']
#Specify the path where you want to download to
img_dir = 'C:/Users/alber/Desktop/UniTn/Data Science/Second_Semester/Machine_Learning/Competition/Img_scarp'
#Repeat extracting for every words
for word in search_words:
    dir_path = img_dir + word
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    #Make word suitable for URL
    urlKeyword = parse.quote(word)
    #Create url with target word
    url = 'https://www.google.com/search?hl=jp&q=' + urlKeyword + '&btnG=Google+Search&tbs=0&safe=off&tbm=isch'
    # headers is necessary when you send request
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.212 Safari/537.36'}
    #Request is made with url and headers
    ## response = request.get(url, headers)
    request = Request(url=url, headers=headers)
    ##page = req.urlopen(request)
    # page that you reveived into utf-8
    
    html = PageElement.read().decode('utf-8')
    #### html = response.text 
    #Use BeutifulSoup, “html.parser”. HTML object is available in Python.
    html = BeautifulSoup(html, 'html.parser')
    # .rg_meta.notranslate is for extracting data without permission(not sure)
    elems = html.select('.rg_meta.notranslate')
    counter = 0
    error_counter = 0
    for ele in elems:
        ele = ele.contents[0].replace(',').split(',')
        eledict = dict()
        #imageURL is indicated here.
        for e in ele:
            num = e.find(':')
            eledict[e[0:num]] = e[num+1:]
            imageURL = eledict['ou']
#URL retrieve: extract imageURL from file_path
#Try catching image, if error occurs, execute except program
    try:
        file_path = dir_path + '/' + str(counter)+ '.jpg'
        request.urlretrieve(imageURL, file_path)
        counter += 1
    except Exception as e:
        error_counter += 1
        if counter == 1:
            print('Start downloading', word)
        if counter==10:
            break
        print('{} errors occur out of '.format(counter, error_counter))'''