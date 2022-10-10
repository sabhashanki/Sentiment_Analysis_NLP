from flask import Flask, request, render_template, jsonify
import requests
import pickle
import pandas as pd
from wordcloud import WordCloud
import re
from textblob import Word
import seaborn as sns
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import string

rfmodel = pickle.load(open('sentiment.pkl','rb'))
# NLP model 

df_train = pd.read_csv('train.txt', delimiter = ';', names = ['text','label'])
df_val = pd.read_csv('test.txt', delimiter = ';', names = ['text','label'])
df = pd.concat([df_train, df_val])
df.reset_index(inplace = True, drop = True)

encoder_list = {
    'surprise' : 1,
    'love' : 1,
    'joy' : 1, 
    'fear' : 0,
    'anger' : 0,
    'sadness' : 0
}
def label_encoder(sample):
    for label, encoder in encoder_list.items():
        if sample == label:
            return encoder

df.label = df.label.apply(lambda x : label_encoder(x))

punc = string.punctuation
allstopwords = stopwords.words('english')

def conversion(sentence):
    lem_words = []
    for row in sentence:
        line = "".join([i for i in row if i not in punc])
        line = [word for word in line.split() if word not in allstopwords]
        line = " ".join([Word(i).lemmatize('v') for i in line])
        lem_words.append(''.join(str(x) for x in line))
    return lem_words

data = conversion(df.text)

vector = CountVectorizer(ngram_range = (1,2))
vector_matrix = vector.fit_transform(data)

def sentiment_prediction(input):
    converted_input = conversion(input)
    vectorized_input = vector.transform(converted_input)
    prediction = rfmodel.predict(vectorized_input)
    return prediction

# Flask Application

app = Flask(__name__)

@app.route('/', methods = ['GET'])
def home():
    return render_template('home.html')

@app.route('/predict', methods = ['POST'])
def predict():

    tweet = request.form['tweet']
    output = sentiment_prediction([tweet])[0]
    if output == 0:
        output = 'Negative Statement'
    elif output == 1:
        output = 'Postitive Statement'
    else:
        output = 'Invalid Statement'
        
    return render_template('home.html', prediction_text = output)

# Driver Code
if __name__ == '__main__':
    app.run(debug = True)