'''
This is a program, which takes two artist-names as input, scrapes 10 of their respective lyrics from lyrics.com,
fits a Naive Bays model on the lyrics corpus and then predicts if some user-input lyric was written by
either one of the given artists.

By: Valentin Lorenzen
Context: Spiced Data Science Bootcamp - July 2022
'''

import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB

import requests
import re
import os

from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer

import sys
import argparse

import nltk
from nltk.tokenize import TreebankWordTokenizer
from nltk.stem import WordNetLemmatizer

import csv
import pickle

from rich.progress import track
from time import sleep

from art import tprint


# print logo
print ("\n\n\n" + '\033[94m')
tprint('LYRICS2')

# welcome message
print ('\033[94m' + "Hi.\n\nThis is a little program which takes two artists, scrapes their\nlyrics from lyrics.com, takes some lyric as an\ninput and predicts whether that lyric was written by either one of them.\n\nType", '\033[1m', "exit" + '\033[0m' + '\033[94m'  + "  to exit\n\n")


# get artists from user
first_artist = input('\033[94m' + "Enter first artist: " + '\033[95m')
second_artist = input('\033[94m' + "Enter second artist: " + '\033[95m')
print ("\n")

# change whitespace to "-" in first artist name
first_artist_html = ""
for x in first_artist:
    if x == " ":
        first_artist_html = first_artist_html + "-"
    else:
        first_artist_html = first_artist_html + x

# download the html of the artist page on lyrics.com
all_of_first = requests.get('https://www.lyrics.com/artist/' + first_artist_html).text

# find all the lyric links in the scraped html
first_artist_songlinks = re.findall(pattern='a href="(/lyric/.{9})', string=all_of_first)

# delete duplicates
first_artist_songlinks = list(dict.fromkeys(first_artist_songlinks))

# create list of lyric links
first_artist_songlinks_final = ["http://www.lyrics.com/" + x + "/" for x in first_artist_songlinks]

# only keep first 10 lyric links
first_artist_songlinks_final = first_artist_songlinks_final[0:10]

# create folder with artist name
if not os.path.exists(first_artist):
    os.mkdir(first_artist)

# scrape lyric-page html
def scrape_data_first(link_input):
    sleep(0.1)
    song_number = re.findall(pattern='(\d{6,9})', string=link_input)
    f = open(f"{first_artist}/{song_number}.txt", "w")
    song_html = requests.get(link_input).text
    f.write(song_html)
    f.close()

# display progress bar    
counter_first = 0
for _ in track(range(10), description='[green]Scraping first artist'):
    scrape_data_first(first_artist_songlinks_final[counter_first])
    counter_first += 1

# open scraped lyric text files and build a corpus
first_artist_lyrics = []
for fn in os.listdir(first_artist):
     text = open(f"{first_artist}/{fn}").read()
     first_artist_soup = BeautifulSoup(text, 'html.parser')
     lyric = first_artist_soup.find_all('pre',{"class":"lyric-body"})
     first_artist_lyrics.append(lyric)

# define cleaning function
cleaner = re.compile('<.*?>') 
def cleanhtml(raw_html):
  cleantext = re.sub(cleaner, '', raw_html)
  return cleantext

# clean lyrics
first_artist_lyrics_clean = []
for x in first_artist_lyrics:
    x_clean = cleanhtml(str(x))
    x_clean = re.sub('[^A-Za-z\s]+', '', x_clean)
    first_artist_lyrics_clean.append(x_clean.lower().replace('\n', ' ').replace("\'", ""))


# repeat the steps for the second artist
second_artist_html = ""
for x in second_artist:
    if x == " ":
        second_artist_html = second_artist_html + "-"
    else:
        second_artist_html = second_artist_html + x
all_of_second = requests.get('https://www.lyrics.com/artist/' + second_artist_html).text
second_artist_songlinks = re.findall(pattern='a href="(/lyric/.{9})', string=all_of_second)
second_artist_songlinks = list(dict.fromkeys(second_artist_songlinks))
second_artist_songlinks_final = ["http://www.lyrics.com/" + x for x in second_artist_songlinks]
second_artist_songlinks_final = second_artist_songlinks_final[0:10]
if not os.path.exists(second_artist):
    os.mkdir(second_artist)
def scrape_data_second(link_input):
    sleep(0.1)
    song_number = re.findall(pattern='(\d{6,9})', string=link_input)
    f = open(f"{second_artist}/{song_number}.txt", "w")
    song_html = requests.get(link_input).text
    f.write(song_html)
    f.close()
counter_second = 0
for _ in track(range(10), description='[green]Scraping second artist'):
    scrape_data_second(second_artist_songlinks_final[counter_second])
    counter_second += 1
second_artist_lyrics = []
for fn in os.listdir(second_artist):
     text = open(f"{second_artist}/{fn}").read()
     second_artist_soup = BeautifulSoup(text, 'html.parser')
     lyric = second_artist_soup.find_all('pre',{"class":"lyric-body"})
     second_artist_lyrics.append(lyric)
second_artist_lyrics_clean = []
for x in second_artist_lyrics:
    x_clean = cleanhtml(str(x))
    x_clean = re.sub('[^A-Za-z\s]+', '', x_clean)
    second_artist_lyrics_clean.append(x_clean.lower().replace('\n', ' ').replace("\'", ""))



# create corpus of both artists lyrics
corpus = first_artist_lyrics_clean + second_artist_lyrics_clean

# define tokenizer and lemmatizer
tokenizer = TreebankWordTokenizer()
lemmatizer = WordNetLemmatizer()

# lemmatize and tokenize the corpus
clean_corpus = []
for doc in corpus:
    tokens = tokenizer.tokenize(text=doc)
    clean_doc = " ".join(lemmatizer.lemmatize(token) for token in tokens)
    clean_corpus.append(clean_doc)

print ("\n")

# import and download stopwords
from nltk.corpus import stopwords
STOPWORDS = stopwords.words('english')

# create labels
LABELS = [first_artist] * 10 + [second_artist] * 10

# create pipeline
steps = [
          ('tf-idf', TfidfVectorizer(stop_words=STOPWORDS)),        
          ('LR', MultinomialNB())
        ]
pipeline = Pipeline(steps)

# fit pipeline on data
pipeline.fit(clean_corpus, LABELS)

# function giving artist and probability
def give_artist(lyric):
    print ('\033[96m')
    if not lyric:
        return ("\n\nYou did not give an input.\n\n" + '\033[94m')
    probab = round(100*(max(pipeline.predict_proba([lyric])[0])))
    who_wrote = pipeline.predict([lyric])[0]
    if probab == 50:
        return ("\n\nI am sorry, the model cannot decide, try some other lyrics.\n\n" + '\033[94m')
    func_return = "\n'" +  lyric + "' was probably written by:\n--- " + who_wrote + "\nWith a certainty of: " + str(probab) + "%\n"
    return (func_return + '\033[94m')

# get user input, run function and give output
while True:
    name = input('\033[94m' + "Enter your lyric: " + '\033[95m')
    if name == "exit":
        print ("\nExiting...\n")
        break
    print(give_artist(name))


