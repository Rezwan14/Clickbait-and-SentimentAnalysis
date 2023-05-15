import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords
stopwords = nltk.corpus.stopwords.words("english")
import string
from urllib.request import urlopen
from bs4 import BeautifulSoup
import requests
from raw_data import *
from statistics import mean
import seaborn as sns
import matplotlib.pyplot as plt

#a dictionary containint all skysports articles:
ss_articles = {ss1, ss2, ss3, ss4, ss5, ss6, ss7, ss8, ss9, ss10}

#a dictionary containing all espn articles:
espn_articles = {espn1, espn2, espn3, espn4, espn5, espn6, espn7, espn8, espn9, espn10}

#a dictionary containing cbs articles
cbs_articles = {cbs1, cbs2, cbs3, cbs4, cbs5, cbs6, cbs7, cbs8, cbs9, cbs10}

#a dictionary containing all control websites articles
cw_articles = {cw1, cw2, cw3, cw4, cw5, cw6, cw7, cw8, cw9, cw10}

#a dictionary containing all of our articles
all_articles = {ss1, ss2, ss3, ss4, ss5, ss6, ss7, ss8, ss9, ss10,
                espn1, espn2, espn3, espn4, espn5, espn6, espn7, espn8, espn9, espn10,
                cbs1, cbs2, cbs3, cbs4, cbs5, cbs6, cbs7, cbs8, cbs9, cbs10,
                cw1, cw2, cw3, cw4, cw5, cw6, cw7, cw8, cw9, cw10}

ss_lower = {ss1.lower(), ss2.lower(), ss3.lower(), ss4.lower(), ss5.lower(), 
               ss6.lower(), ss7.lower(), ss8.lower(), ss9.lower(), ss10.lower()}

#a dictionary containing all espn articles:
espn_articles = {espn1.lower(), espn2.lower(), espn3.lower(), espn4.lower(), 
                 espn5.lower(), espn6.lower(), espn7.lower(), espn8.lower(), espn9.lower(), espn10.lower()}

#a dictionary containing cbs articles
cbs_articles = {cbs1.lower(), cbs2.lower(), cbs3.lower(), cbs4.lower(), cbs5.lower(), 
                cbs6.lower(), cbs7.lower(), cbs8.lower(), cbs9.lower(), cbs10.lower()}

#a dictionary containing all control websites articles
cw_articles = {cw1.lower(), cw2.lower(), cw3.lower(), cw4.lower(), cw5.lower(), 
               cw6.lower(), cw7.lower(), cw8.lower(), cw9.lower(), cw10.lower()}

#a dictionary containing all of our articles
all_articles = {ss1.lower(), ss2.lower(), ss3.lower(), ss4.lower(), ss5.lower(), 
               ss6.lower(), ss7.lower(), ss8.lower(), ss9.lower(), ss10.lower(),
                espn1.lower(), espn2.lower(), espn3.lower(), espn4.lower(), 
                 espn5.lower(), espn6.lower(), espn7.lower(), espn8.lower(), espn9.lower(), espn10.lower(),
                cbs1.lower(), cbs2.lower(), cbs3.lower(), cbs4.lower(), cbs5.lower(), 
                cbs6.lower(), cbs7.lower(), cbs8.lower(), cbs9.lower(), cbs10.lower(),
                cw1.lower(), cw2.lower(), cw3.lower(), cw4.lower(), cw5.lower(), 
               cw6.lower(), cw7.lower(), cw8.lower(), cw9.lower(), cw10.lower()}


import string
regular_punct = list(string.punctuation)
def remove_punctuation(text,punct_list):
    for punc in punct_list:
        if punc in text:
            text = text.replace(punc, ' ')
    return text.strip()
#test to see if our function is working
remove_punctuation(ss10.lower(),regular_punct)

#Remove punctuation from all the skysports articles
ss1_punct = remove_punctuation(ss1.lower(),regular_punct)
ss2_punct = remove_punctuation(ss2.lower(),regular_punct)
ss3_punct = remove_punctuation(ss3.lower(),regular_punct)
ss4_punct = remove_punctuation(ss4.lower(),regular_punct)
ss5_punct = remove_punctuation(ss5.lower(),regular_punct)
ss6_punct = remove_punctuation(ss6.lower(),regular_punct)
ss7_punct = remove_punctuation(ss7.lower(),regular_punct)
ss8_punct = remove_punctuation(ss8.lower(),regular_punct)
ss9_punct = remove_punctuation(ss9.lower(),regular_punct)
ss10_punct = remove_punctuation(ss10.lower(),regular_punct)

#remove punctuation from all the espn articles
espn1_punct = remove_punctuation(espn1.lower(),regular_punct)
espn2_punct = remove_punctuation(espn2.lower(),regular_punct)
espn3_punct = remove_punctuation(espn3.lower(),regular_punct)
espn4_punct = remove_punctuation(espn4.lower(),regular_punct)
espn5_punct = remove_punctuation(espn5.lower(),regular_punct)
espn6_punct = remove_punctuation(espn6.lower(),regular_punct)
espn7_punct = remove_punctuation(espn7.lower(),regular_punct)
espn8_punct = remove_punctuation(espn8.lower(),regular_punct)
espn9_punct = remove_punctuation(espn9.lower(),regular_punct)
espn10_punct = remove_punctuation(espn10.lower(),regular_punct)

#remove punctuation from all the cbs articles
cbs1_punct = remove_punctuation(cbs1.lower(),regular_punct)
cbs2_punct = remove_punctuation(cbs2.lower(),regular_punct)
cbs3_punct = remove_punctuation(cbs3.lower(),regular_punct)
cbs4_punct = remove_punctuation(cbs4.lower(),regular_punct)
cbs5_punct = remove_punctuation(cbs5.lower(),regular_punct)
cbs6_punct = remove_punctuation(cbs6.lower(),regular_punct)
cbs7_punct = remove_punctuation(cbs7.lower(),regular_punct)
cbs8_punct = remove_punctuation(cbs8.lower(),regular_punct)
cbs9_punct = remove_punctuation(cbs9.lower(),regular_punct)
cbs10_punct = remove_punctuation(cbs10.lower(),regular_punct)

#remove punctuation from all the control website articles
cw1_punct = remove_punctuation(cw1.lower(),regular_punct)
cw2_punct = remove_punctuation(cw2.lower(),regular_punct)
cw3_punct = remove_punctuation(cw3.lower(),regular_punct)
cw4_punct = remove_punctuation(cw4.lower(),regular_punct)
cw5_punct = remove_punctuation(cw5.lower(),regular_punct)
cw6_punct = remove_punctuation(cw6.lower(),regular_punct)
cw7_punct = remove_punctuation(cw7.lower(),regular_punct)
cw8_punct = remove_punctuation(cw8.lower(),regular_punct)
cw9_punct = remove_punctuation(cw9.lower(),regular_punct)
cw10_punct = remove_punctuation(cw10.lower(),regular_punct)


#try a test method to see if task is working properly
from nltk.tokenize import word_tokenize

#Tokenize all our words in skysports:
ss1_tokenize = word_tokenize(ss1_punct)
ss2_tokenize = word_tokenize(ss2_punct)
ss3_tokenize = word_tokenize(ss3_punct)
ss4_tokenize = word_tokenize(ss4_punct)
ss5_tokenize = word_tokenize(ss5_punct)
ss6_tokenize = word_tokenize(ss6_punct)
ss7_tokenize = word_tokenize(ss7_punct)
ss8_tokenize = word_tokenize(ss8_punct)
ss9_tokenize = word_tokenize(ss9_punct)
ss10_tokenize = word_tokenize(ss10_punct)

#Tokenize all our words in espn:
espn1_tokenize = word_tokenize(espn1_punct)
espn2_tokenize = word_tokenize(espn2_punct)
espn3_tokenize = word_tokenize(espn3_punct)
espn4_tokenize = word_tokenize(espn4_punct)
espn5_tokenize = word_tokenize(espn5_punct)
espn6_tokenize = word_tokenize(espn6_punct)
espn7_tokenize = word_tokenize(espn7_punct)
espn8_tokenize = word_tokenize(espn8_punct)
espn9_tokenize = word_tokenize(espn9_punct)
espn10_tokenize = word_tokenize(espn10_punct)

#Tokenize all our words in cbs:
cbs1_tokenize = word_tokenize(cbs1_punct)
cbs2_tokenize = word_tokenize(cbs2_punct)
cbs3_tokenize = word_tokenize(cbs3_punct)
cbs4_tokenize = word_tokenize(cbs4_punct)
cbs5_tokenize = word_tokenize(cbs5_punct)
cbs6_tokenize = word_tokenize(cbs6_punct)
cbs7_tokenize = word_tokenize(cbs7_punct)
cbs8_tokenize = word_tokenize(cbs8_punct)
cbs9_tokenize = word_tokenize(cbs9_punct)
cbs10_tokenize = word_tokenize(cbs10_punct)

#Tokenize all our words in controlwebsite:
cw1_tokenize = word_tokenize(cw1_punct)
cw2_tokenize = word_tokenize(cw2_punct)
cw3_tokenize = word_tokenize(cw3_punct)
cw4_tokenize = word_tokenize(cw4_punct)
cw5_tokenize = word_tokenize(cw5_punct)
cw6_tokenize = word_tokenize(cw6_punct)
cw7_tokenize = word_tokenize(cw7_punct)
cw8_tokenize = word_tokenize(cw8_punct)
cw9_tokenize = word_tokenize(cw9_punct)
cw10_tokenize = word_tokenize(cw10_punct)


from nltk.corpus import stopwords
stop_words_array = set(stopwords.words('english'))
#remove stopwords for skysports:
ss1_s =  []
for w in ss1_tokenize:
    if w not in stop_words_array:
        ss1_s.append(w)
ss2_s =  []
for w in ss2_tokenize:
    if w not in stop_words_array:
        ss2_s.append(w)
ss3_s =  []
for w in ss3_tokenize:
    if w not in stop_words_array:
        ss3_s.append(w)
ss4_s =  []
for w in ss4_tokenize:
    if w not in stop_words_array:
        ss4_s.append(w)
ss5_s =  []
for w in ss5_tokenize:
    if w not in stop_words_array:
        ss5_s.append(w)
ss6_s =  []
for w in ss6_tokenize:
    if w not in stop_words_array:
        ss6_s.append(w)
ss7_s =  []
for w in ss7_tokenize:
    if w not in stop_words_array:
        ss7_s.append(w)
ss8_s =  []
for w in ss8_tokenize:
    if w not in stop_words_array:
        ss8_s.append(w)
ss9_s =  []
for w in ss9_tokenize:
    if w not in stop_words_array:
        ss9_s.append(w)
ss10_s =  []
for w in ss10_tokenize:
    if w not in stop_words_array:
        ss10_s.append(w)
#remove stopwords from espn articles
espn1_s =  []
for w in espn1_tokenize:
    if w not in stop_words_array:
        espn1_s.append(w)
espn2_s =  []
for w in espn2_tokenize:
    if w not in stop_words_array:
        espn2_s.append(w)
espn3_s =  []
for w in espn3_tokenize:
    if w not in stop_words_array:
        espn3_s.append(w)
espn4_s =  []
for w in espn4_tokenize:
    if w not in stop_words_array:
        espn4_s.append(w)
espn5_s =  []
for w in espn5_tokenize:
    if w not in stop_words_array:
        espn5_s.append(w)
espn6_s =  []
for w in espn6_tokenize:
    if w not in stop_words_array:
        espn6_s.append(w)
espn7_s =  []
for w in espn7_tokenize:
    if w not in stop_words_array:
        espn7_s.append(w)
espn8_s =  []
for w in espn8_tokenize:
    if w not in stop_words_array:
        espn8_s.append(w)
espn9_s =  []
for w in espn9_tokenize:
    if w not in stop_words_array:
        espn9_s.append(w)
espn10_s =  []
for w in espn10_tokenize:
    if w not in stop_words_array:
        espn10_s.append(w)
#remove stopwords from cbs articles
cbs1_s =  []
for w in cbs1_tokenize:
    if w not in stop_words_array:
        cbs1_s.append(w)
cbs2_s =  []
for w in cbs2_tokenize:
    if w not in stop_words_array:
        cbs2_s.append(w)
cbs3_s =  []
for w in cbs3_tokenize:
    if w not in stop_words_array:
        cbs3_s.append(w)
cbs4_s =  []
for w in cbs4_tokenize:
    if w not in stop_words_array:
        cbs4_s.append(w)
cbs5_s =  []
for w in cbs5_tokenize:
    if w not in stop_words_array:
        cbs5_s.append(w)
cbs6_s =  []
for w in cbs6_tokenize:
    if w not in stop_words_array:
        cbs6_s.append(w)
cbs7_s =  []
for w in cbs7_tokenize:
    if w not in stop_words_array:
        cbs7_s.append(w)
cbs8_s =  []
for w in cbs8_tokenize:
    if w not in stop_words_array:
        cbs8_s.append(w)
cbs9_s =  []
for w in cbs9_tokenize:
    if w not in stop_words_array:
        cbs9_s.append(w)
cbs10_s =  []
for w in cbs10_tokenize:
    if w not in stop_words_array:
        cbs10_s.append(w)
#remove stopwords from every control website article:
cw1_s =  []
for w in cw1_tokenize:
    if w not in stop_words_array:
        cw1_s.append(w)
cw2_s =  []
for w in cw2_tokenize:
    if w not in stop_words_array:
        cw2_s.append(w)
cw3_s =  []
for w in cw3_tokenize:
    if w not in stop_words_array:
        cw3_s.append(w)
cw4_s =  []
for w in cw4_tokenize:
    if w not in stop_words_array:
        cw4_s.append(w)
cw5_s =  []
for w in cw5_tokenize:
    if w not in stop_words_array:
        cw5_s.append(w)
cw6_s =  []
for w in cw6_tokenize:
    if w not in stop_words_array:
        cw6_s.append(w)
cw7_s =  []
for w in cw7_tokenize:
    if w not in stop_words_array:
        cw7_s.append(w)
cw8_s =  []
for w in cw8_tokenize:
    if w not in stop_words_array:
        cw8_s.append(w)
cw9_s =  []
for w in cw9_tokenize:
    if w not in stop_words_array:
        cw9_s.append(w)
cw10_s =  []
for w in cw10_tokenize:
    if w not in stop_words_array:
        cw10_s.append(w)

#list of hyperbole words
hyperbole_words = [
    'abundant',
    'absolute',
    'amazing',
    'amazed',
    'amazingly',
    'astonishing',
    'astounding',
    'awesome',
    'awful',
    'awfully',
    'beautiful',
    'beastly',
    'big',
    'bust',
    'bizarre',
    'blinding',
    'boiling',
    'bold',
    'bombastic',
    'breeze',
    'breathtaking',
    'brilliant',
    'burning',
    'buzzing',
    'catastrophic',
    'celestial',
    'charming',
    'colossal',
    'cool',
    'crazy',
    'crushing',
    'dazzling',
    'deadly',
    'delicious',
    'delightful',
    'dissapoint',
    'dissapointing',
    'demonic',
    'destructive',
    'devilish',
    'dramatic',
    'dreamy',
    'dreadful',
    'dynamic',
    'elegant',
    'elite',
    'enchanted',
    'enormous',
    'epic',
    'eternal',
    'excellent',
    'exciting',
    'explosive',
    'exquisite',
    'electric',
    'extraordinary',
    'fabulous',
    'fantastic',
    'fault',
    'fearsome',
    'ferocious',
    'fiery',
    'flaming',
    'flawless',
    'flooded',
    'flowing',
    'fluffy',
    'formidable',
    'foolish',
    'frozen',
    'furious',
    'galactic',
    'gargantuan',
    'giant',
    'glamorous',
    'glittering',
    'glistening',
    'glorious',
    'gnarly',
    'golden',
    'gorgeous',
    'grandiose',
    'gruesome',
    'great',
    'greatest',
    'heavenly',
    'heavy',
    'hellish',
    'heroic',
    'hideous',
    'hilarious',
    'horrible',
    'horrific',
    'huge',
    'hulking',
    'humongous',
    'hysterical',
    'iconic',
    'immaculate',
    'impressive',
    'incredible',
    'indescribable',
    'infinite',
    'insane',
    'intense',
    'intriguing',
    'invincible',
    'jagged',
    'jubilant',
    'legendary',
    'lightning',
    'limitless',
    'luminous',
    'magical',
    'majestic',
    'massive',
    'melodic',
    'menacing',
    'merciless',
    'mighty',
    'monster',
    'monstrous',
    'mystical',
    'mythical',
    'nebulous',
    'nightmarish',
    'noble',
    'ominous',
    'outstanding',
    'overrated',
    'overwhelming',
    'pandemonic',
    'peaceful',
    'perfect',
    'phenomenal',
    'platinum',
    'pleased',
    'pleasing',
    'poisonous',
    'powerful',
    'primal',
    'prodigious',
    'profound',
    'pulsating',
    'pulsing',
    'pure',
    'predict',
    'predicts',
    'prediction',
    'radiant',
    'raging',
    'rampaging',
    'ravenous',
    'remarkable',
    'resplendent',
    'roaring',
    'robust',
    'savage',
    'scorching',
    'scrumptious',
    'searing',
    'sensational',
    'shambles',
    'shame',
    'shambolic',
    'shocking',
    'show-stopping',
    'shuddering',
    'sickening',
    'sizzling',
    'specimen',
    'spectacular',
    'spellbinding',
    'spine-tingling',
    'splendid',
    'staggering',
    'star',
    'stellar',
    'stunning',
    'sublime',
    'shine',
    'shined',
    'superb',
    'supreme',
    'sweltering',
    'tantalizing',
    'tear-jerking',
    'terrific',
    'thundering',
    'titanic',
    'trash',
    'top-notch',
    'tremendous',
    'unbelievable',
    'unforgettable',
    'unimaginable',
    'unprecedented',
    'underwhelm',
    'underwhelming',
    'underrated',
    'unreal',
    'unrivaled',
    'unstoppable',
    'uplifting',
    'vast',
    'venomous',
    'vibrant',
    'victorious',
    'violent',
    'virtuosic',
    'visionary',
    'vivid',
    'volcanic',
    'wicked',
    'wild',
    'wondrous',
    'world-class',
    'wow-worthy',
    'zealous'
]

ss1_hyperbole = sum(ss1_s.count(x) for x in (hyperbole_words))
ss1_percentage = (ss1_hyperbole / len(ss1_s)) * 10
ss1_score = round(ss1_percentage, 2)


ss2_hyperbole = sum(ss2_s.count(x) for x in (hyperbole_words))
ss2_percentage = (ss2_hyperbole / len(ss2_s)) * 10
ss2_score = round(ss2_percentage, 2)

ss3_hyperbole = sum(ss3_s.count(x) for x in (hyperbole_words))
ss3_percentage = (ss3_hyperbole / len(ss3_s)) * 10
ss3_score = round(ss3_percentage, 2)

ss4_hyperbole = sum(ss4_s.count(x) for x in (hyperbole_words))
ss4_percentage = (ss4_hyperbole / len(ss4_s)) * 10
ss4_score = round(ss4_percentage, 2)

ss5_hyperbole = sum(ss5_s.count(x) for x in (hyperbole_words))
ss5_percentage = (ss5_hyperbole / len(ss5_s)) * 10
ss5_score = round(ss5_percentage, 2)

ss6_hyperbole = sum(ss6_s.count(x) for x in (hyperbole_words))
ss6_percentage = (ss6_hyperbole / len(ss6_s)) * 10
ss6_score = round(ss6_percentage, 2)

ss7_hyperbole = sum(ss7_s.count(x) for x in (hyperbole_words))
ss7_percentage = (ss7_hyperbole / len(ss7_s)) * 10
ss7_score = round(ss7_percentage, 2)

ss8_hyperbole = sum(ss8_s.count(x) for x in (hyperbole_words))
ss8_percentage = (ss8_hyperbole / len(ss8_s)) * 10
ss8_score = round(ss8_percentage, 2)

ss9_hyperbole = sum(ss9_s.count(x) for x in (hyperbole_words))
ss9_percentage = (ss9_hyperbole / len(ss9_s)) * 10
ss9_score = round(ss9_percentage, 2)

ss10_hyperbole = sum(ss10_s.count(x) for x in (hyperbole_words))
ss10_percentage = (ss10_hyperbole / len(ss10_s)) * 10
ss10_score = round(ss10_percentage, 2)

espn1_hyperbole = sum(espn1_s.count(x) for x in (hyperbole_words))
espn1_percentage = (espn1_hyperbole / len(espn1_s)) * 10
espn1_score = round(espn1_percentage, 2)

espn2_hyperbole = sum(espn2_s.count(x) for x in (hyperbole_words))
espn2_percentage = (espn2_hyperbole / len(espn2_s)) * 10
espn2_score = round(espn2_percentage, 2)

espn3_hyperbole = sum(espn3_s.count(x) for x in (hyperbole_words))
espn3_percentage = (espn3_hyperbole / len(espn3_s)) * 10
espn3_score = round(espn3_percentage, 2)

espn4_hyperbole = sum(espn4_s.count(x) for x in (hyperbole_words))
espn4_percentage = (espn4_hyperbole / len(espn4_s)) * 10
espn4_score = round(espn4_percentage, 2)

espn5_hyperbole = sum(espn5_s.count(x) for x in (hyperbole_words))
espn5_percentage = (espn5_hyperbole / len(espn5_s)) * 10
espn5_score = round(espn5_percentage, 2)

espn6_hyperbole = sum(espn6_s.count(x) for x in (hyperbole_words))
espn6_percentage = (espn6_hyperbole / len(espn6_s)) * 10
espn6_score = round(espn6_percentage, 2)

espn7_hyperbole = sum(espn7_s.count(x) for x in (hyperbole_words))
espn7_percentage = (espn7_hyperbole / len(espn7_s)) * 10
espn7_score = round(espn7_percentage, 2)

espn8_hyperbole = sum(espn8_s.count(x) for x in (hyperbole_words))
espn8_percentage = (espn8_hyperbole / len(espn8_s)) * 10
espn8_score = round(espn8_percentage, 2)

espn9_hyperbole = sum(espn9_s.count(x) for x in (hyperbole_words))
espn9_percentage = (espn9_hyperbole / len(espn9_s)) * 10
espn9_score = round(espn9_percentage, 2)

espn10_hyperbole = sum(espn10_s.count(x) for x in (hyperbole_words))
espn10_percentage = (espn10_hyperbole / len(espn10_s)) * 10
espn10_score = round(espn10_percentage, 2)

cbs1_hyperbole = sum(cbs1_s.count(x) for x in (hyperbole_words))
cbs1_percentage = (cbs1_hyperbole / len(cbs1_s)) * 10
cbs1_score = round(cbs1_percentage, 2)

cbs2_hyperbole = sum(cbs2_s.count(x) for x in (hyperbole_words))
cbs2_percentage = (cbs2_hyperbole / len(cbs2_s)) * 10
cbs2_score = round(cbs2_percentage, 2)

cbs3_hyperbole = sum(cbs3_s.count(x) for x in (hyperbole_words))
cbs3_percentage = (cbs3_hyperbole / len(cbs3_s)) * 10
cbs3_score = round(cbs3_percentage, 2)

cbs4_hyperbole = sum(cbs4_s.count(x) for x in (hyperbole_words))
cbs4_percentage = (cbs4_hyperbole / len(cbs4_s)) * 10
cbs4_score = round(cbs4_percentage, 2)

cbs5_hyperbole = sum(cbs5_s.count(x) for x in (hyperbole_words))
cbs5_percentage = (cbs5_hyperbole / len(cbs5_s)) * 10
cbs5_score = round(cbs5_percentage, 2)

cbs6_hyperbole = sum(cbs6_s.count(x) for x in (hyperbole_words))
cbs6_percentage = (cbs6_hyperbole / len(cbs6_s)) * 10
cbs6_score = round(cbs6_percentage, 2)

cbs7_hyperbole = sum(cbs7_s.count(x) for x in (hyperbole_words))
cbs7_percentage = (cbs7_hyperbole / len(cbs7_s)) * 10
cbs7_score = round(cbs7_percentage, 2)

cbs8_hyperbole = sum(cbs8_s.count(x) for x in (hyperbole_words))
cbs8_percentage = (cbs8_hyperbole / len(cbs8_s)) * 10
cbs8_score = round(cbs8_percentage, 2)

cbs9_hyperbole = sum(cbs9_s.count(x) for x in (hyperbole_words))
cbs9_percentage = (cbs9_hyperbole / len(cbs9_s)) * 10
cbs9_score = round(cbs9_percentage, 2)

cbs10_hyperbole = sum(cbs10_s.count(x) for x in (hyperbole_words))
cbs10_percentage = (cbs10_hyperbole / len(cbs10_s)) * 10
cbs10_score = round(cbs10_percentage, 2)

cw1_hyperbole = sum(cw1_s.count(x) for x in (hyperbole_words))
cw1_percentage = (cw1_hyperbole / len(cw1_s)) * 10
cw1_score = round(cw1_percentage, 2)


cw2_hyperbole = sum(cw2_s.count(x) for x in (hyperbole_words))
cw2_percentage = (cw2_hyperbole / len(cw2_s)) * 10
cw2_score = round(cw2_percentage, 2)

cw3_hyperbole = sum(cw3_s.count(x) for x in (hyperbole_words))
cw3_percentage = (cw3_hyperbole / len(cw3_s)) * 10
cw3_score = round(cw3_percentage, 2)

cw4_hyperbole = sum(cw4_s.count(x) for x in (hyperbole_words))
cw4_percentage = (cw4_hyperbole / len(cw4_s)) * 10
cw4_score = round(cw4_percentage, 2)

cw5_hyperbole = sum(cw5_s.count(x) for x in (hyperbole_words))
cw5_percentage = (cw5_hyperbole / len(cw5_s)) * 10
cw5_score = round(cw5_percentage, 2)

cw6_hyperbole = sum(cw6_s.count(x) for x in (hyperbole_words))
cw6_percentage = (cw6_hyperbole / len(cw6_s)) * 10
cw6_score = round(cw6_percentage, 2)

cw7_hyperbole = sum(cw7_s.count(x) for x in (hyperbole_words))
cw7_percentage = (cw7_hyperbole / len(cw7_s)) * 10
cw7_score = round(cw7_percentage, 2)

cw8_hyperbole = sum(cw8_s.count(x) for x in (hyperbole_words))
cw8_percentage = (cw8_hyperbole / len(cw8_s)) * 10
cw8_score = round(cw8_percentage, 2)

cw9_hyperbole = sum(cw9_s.count(x) for x in (hyperbole_words))
cw9_percentage = (cw9_hyperbole / len(cw9_s)) * 10
cw9_score = round(cw9_percentage, 2)

cw10_hyperbole = sum(cw10_s.count(x) for x in (hyperbole_words))
cw10_percentage = (cw10_hyperbole / len(cw10_s)) * 10
cw10_score = round(cw10_percentage, 2)

ss_hyperbole_array = [ss1_score, ss2_score, ss3_score, ss4_score, ss5_score, ss6_score, ss7_score, ss8_score, ss9_score, ss10_score]
ss_hyperbole_average = mean(ss_hyperbole_array)
ss_hyperbole_average_final = (round(ss_hyperbole_average,2))
print(ss_hyperbole_average_final)

espn_hyperbole_array = [espn1_score, espn2_score, espn3_score, espn4_score, espn5_score, espn6_score, espn7_score, espn8_score, espn9_score, espn10_score]
espn_hyperbole_average = mean(espn_hyperbole_array)
espn_hyperbole_average_final = (round(espn_hyperbole_average,2))
print(espn_hyperbole_average_final)

cbs_hyperbole_array = [cbs1_score, cbs2_score, cbs3_score, cbs4_score, cbs5_score, cbs6_score, cbs7_score, cbs8_score, cbs9_score, cbs10_score]
cbs_hyperbole_average = mean(cbs_hyperbole_array)
cbs_hyperbole_average_final = (round(cbs_hyperbole_average,2))
print(cbs_hyperbole_average_final)

cw_hyperbole_array = [cw1_score, cw2_score, cw3_score, cw4_score, cw5_score, cw6_score, cw7_score, cw8_score, cw9_score, cw10_score]
cw_hyperbole_average = mean(cw_hyperbole_array)
cw_hyperbole_average_final = (round(cw_hyperbole_average,2))
print(cw_hyperbole_average_final)

import seaborn as sns
import matplotlib.pyplot as plt
y_axis = [ss_hyperbole_average_final, espn_hyperbole_average_final, cbs_hyperbole_average_final, cw_hyperbole_average_final]
x_axis = ['SkySports', 'ESPN', 'CBS SPORTS', 'Control Websites']
plt.bar(x_axis, y_axis)
plt.title('Company Hyperbole Word Usage')
plt.xlabel('Company Name')
plt.ylabel('Hyperbole Score')
max_y_lim = max(y_axis) + 0.20
min_y_lim = min(y_axis) - 0.01
plt.ylim(min_y_lim, max_y_lim)
#plt.figure(figsize(16,9))
plt.show