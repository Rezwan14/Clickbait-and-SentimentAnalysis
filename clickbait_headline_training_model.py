import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from nltk.corpus import stopwords
import string
from nltk.tokenize import word_tokenize
from article_urls import *
from bs4 import BeautifulSoup
import requests

def get_title_of_url(url):
    try:
        response = requests.get(url)
    except requests.exceptions.RequestException:
        return False

    soup = BeautifulSoup(response.content, 'html.parser')
    title_element = soup.find('title')
    if not title_element:
        return False
    else:
        title = title_element.get_text()
        return title
    


# Load the CSV file into a Pandas DataFrame
df = pd.read_csv("clickbait_and_nonclickbait.csv")

# Preprocess the headlines column
stop_words = stopwords.words('english')
punct_marks = string.punctuation

def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Tokenize the text
    tokens = word_tokenize(text)
    # Remove stopwords and nonalphanumeric characters
    tokens = [token for token in tokens if token not in stop_words and token not in punct_marks]
    # Join the tokens back into a string
    text = " ".join(tokens)
    return text

# Preprocessing dataset
df["headline"] = df["headline"].apply(preprocess_text)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df["headline"], df["clickbait"], test_size=0.2, random_state=42)

# Define the machine learning pipeline
text_clf = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', MultinomialNB())
])

# Train the model
text_clf.fit(X_train, y_train)

# Evaluate the model and print accuracy
y_pred = text_clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy of training model: {:.2f}%".format(accuracy * 100))

# def print likelihood of clickbait
def is_clickbait(probability):
    probability = float(probability)
    if(probability < 25):
        print("The headline is very likely not clickbait")
    elif (probability < 50):
        print("The headline is likely not clickbait")
    elif (probability < 75):
        print("The headline is likely clickbait")
    elif (probability >= 75):
        print("The headline is very likely clickbait")

def parse_urls (array_of_urls):
    prob_score = []
    for url in array_of_urls:
        url_title = get_title_of_url(url)
        if url_title == False or url_title == "403 Forbidden": # Error Handling
            continue
        url_prob = text_clf.predict_proba([url_title])
        url_clickbait_prob = "{:.2f}".format(url_prob[0][1] * 100)
        url_clickbait_prob = float(url_clickbait_prob)
        prob_score.append(url_clickbait_prob)
    return prob_score

# # Whole dataset statistics
# dataset_probabilities = text_clf.predict_proba(df["headline"])

# # Compute the average predicted probability
# avg_dataset_prob = np.mean(dataset_probabilities[:, 1])

# print("The average predicted probability of clickbait from the entire dataset: {:.2f}%".format(avg_dataset_prob * 100))

# input_text_headline = input("Enter a headline to see its probability of it being considered clickbait: ")

# predicted_prob = text_clf.predict_proba([input_text_headline])
# clickbait_prob = "{:.2f}".format(predicted_prob[0][1] * 100)
# print(f"The predicted probability of the headline being clickbait is {clickbait_prob}%")
# is_clickbait(clickbait_prob)


# # Input url to grab title
# input_url_headline = input("Enter a url to extract its title to see its probability of it being considered clickbait: ")
# url_title = get_title_of_url(input_url_headline)

# # Repeatedly ask user for title of url if title unable to be extracted/found
# while(url_title == False or url_title == "403 Forbidden"):
#     print("Title of article not found, try a different url")
#     input_url_headline = input("Enter a url to extract its title to see its probability of it being considered clickbait: ")
#     url_title = get_title_of_url(input_url_headline)

# predicted_prob2 = text_clf.predict_proba([url_title])
# clickbait_prob2 = "{:.2f}".format(predicted_prob2[0][1] * 100)
# print(url_title)
# print(f"The predicted probability of the headline being clickbait is {clickbait_prob2}%")
# is_clickbait(clickbait_prob2)

skysports_arr = parse_urls(skysports)
espn_arr = parse_urls(espn)
cbssports_arr = parse_urls(cbssports)
controlwebsites_arr = parse_urls(controlwebsites)

skysports_avg = "{:.2f}".format(sum(skysports_arr) / len(skysports_arr))
espn_avg = "{:.2f}".format(sum(espn_arr) / len(espn_arr))
cbssports_avg = "{:.2f}".format(sum(cbssports_arr) / len(cbssports_arr))
controlwebsites_avg = "{:.2f}".format(sum(controlwebsites_arr) / len(controlwebsites_arr))

print(f"SkySports: {skysports_avg}")
print(f"ESPN: {espn_avg}")
print(f"CBS: {cbssports_avg}")
print(f"control: {controlwebsites_avg}")





