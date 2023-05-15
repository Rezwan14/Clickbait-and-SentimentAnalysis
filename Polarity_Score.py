#use necessary libraries
import nltk
from nltk.tokenize import RegexpTokenizer
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords
stopwords = nltk.corpus.stopwords.words("english")
import string
from urllib.request import urlopen
from bs4 import BeautifulSoup
import requests

##################################################################

url = "https://www.espn.com/soccer/report/_/gameId/656861"
html = urlopen(url).read()
soup = BeautifulSoup(html, features="html.parser")

for script in soup(["script", "style"]):
    script.extract()    # rip it out

# get text
text = soup.get_text()
print(text)

lines = (line.strip() for line in text.splitlines())

chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
# drop blank lines
text = '\n'.join(chunk for chunk in chunks if chunk)

##################################################################

#Data Preparation
#Filter out unnecessary parts of the text
import regex as re
head, sep, tail = text.partition('Terms of UsePrivacy')
stripped = head
print(stripped)

##################################################################

#Data Preparation: Filter out more unnecessary part of text
#There are some unnecessary parts, but this is enough for us to work with
head, sep, tail = stripped.partition('SummaryReportCommentaryStatisticsLine-UpsVideos')
Baseline_text = tail
print(Baseline_text)

##################################################################

#convert text to lowercase
baseline_lower = Baseline_text.lower()
baseline_lower

##################################################################

#remove punctuation from text
import string
regular_punct = list(string.punctuation)
def remove_punctuation(text,punct_list):
    for punc in punct_list:
        if punc in text:
            text = text.replace(punc, ' ')
    return text.strip()

baseline_punct = remove_punctuation(baseline_lower,regular_punct)
baseline_punct

##################################################################

import nltk
from nltk.probability import FreqDist
baseline_words = nltk.word_tokenize(baseline_punct)
baseline_sent = nltk.sent_tokenize(baseline_punct)
print(baseline_words)
print(baseline_sent)
#baseline_fdist = FreqDist(baseline_tokens)
#baseline_fdist

##################################################################

from nltk.corpus import stopwords
set(stopwords.words('english'))

##################################################################

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
stop_words_array = set(stopwords.words('english'))
words = word_tokenize(baseline_punct)
filtered_sentence =  []
for w in words:
    if w not in stop_words_array:
        filtered_sentence.append(w)
print(words)

##################################################################

print("AFTER REMOVING STOPWORDS")
print(filtered_sentence)

##################################################################

import nltk
nltk.downloader.download('vader_lexicon')

##################################################################

from nltk.tokenize.treebank import TreebankWordDetokenizer
Final_Baseline = TreebankWordDetokenizer().detokenize(filtered_sentence)
Final_Baseline

##################################################################

from nltk.sentiment import SentimentIntensityAnalyzer
sia = SentimentIntensityAnalyzer()
sia.polarity_scores(Final_Baseline)

##################################################################

#Lets take a look at our original raw polarity score
sia = SentimentIntensityAnalyzer()
sia.polarity_scores(Baseline_text)