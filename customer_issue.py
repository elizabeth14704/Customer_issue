                                # Consumer Complaint Classification Web App

# Classify Consumer Complaints into Categories Using NLP and Machine Learning
# This project uses Natural Language Processing (NLP) and a K-Nearest Neighbors (KNN) classifier to predict the product category of consumer complaints. The model is deployed using Streamlit for real-time user interaction.



# Importing Libraries and Tools
import streamlit as st
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
import nltk 
import string 

from sklearn.model_selection import train_test_split 
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.metrics import classification_report
from sklearn.naive_bayes import MultinomialNB 
from sklearn.feature_extraction.text import TfidfTransformer 
from sklearn.pipeline import Pipeline
# Imports libraries for text processing (nltk), machine learning (sklearn), data manipulation (pandas, numpy), and visualization (matplotlib, seaborn). Streamlit is used to create the interactive web interface.



# Downloading NLTK Resources
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('punkt') 
# Downloads essential resources from NLTK for tokenization, lemmatization, and stop word removal.



#Loading and Cleaning the Dataset
df = pd.read_csv('C:\\Users\\basil\\Downloads\\complaints\\complaints\\consumercomplaints.csv') 
df = df.head(5000) 
df = df.drop(df[['Unnamed: 0','Sub-issue','Date received','Sub-product','Sub-issue']], axis = 1) 
df = df.dropna(subset = ['Consumer complaint narrative']) 
# Loads a subset (first 5000 entries) of the Consumer Complaint dataset and removes irrelevant or missing columns for simplification.



# Text Preprocessing Function
def preprocess_text(text):
    lemmatizer = WordNetLemmatizer()
    text = text.lower()
    tokens = nltk.word_tokenize(text)
    tokens = [lemmatizer.lemmatize(t) for t in tokens if t not in stopwords.words('english') and t.isalpha()]
    return ' '.join(tokens) 
# Converts text to lowercase, removes stopwords, non-alphabetic characters, and performs lemmatization to normalize the text for model input.



# Apply Preprocessing and Encode Labels
df['clean_text'] = df['Consumer complaint narrative'].apply(preprocess_text)  
df['label'] = df['Product'].astype('category').cat.codes
label_map = dict(enumerate(df['Product'].astype('category').cat.categories))
# Applies text preprocessing and encodes product labels into numeric format. A label_map dictionary is also created for interpretation of predictions.



# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(df['clean_text'], df['label'], test_size =0.2, random_state = 100)  
# Splits the data into training and test sets (80/20) to evaluate model performance.



# Text Vectorization and TF-IDF Transformation
vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

tfid = TfidfTransformer()
new_X_train = tfid.fit_transform(X_train)
new_X_test = tfid.transform(X_test)  
# Converts text into numeric format using Bag of Words and then weights it using TF-IDF, improving classification performance by reducing the influence of common words.



# Train and Evaluate KNN Model
knc = KNeighborsClassifier(n_neighbors= 5)
knc.fit(new_X_train, y_train)
predict = knc.predict(new_X_test)
print(classification_report(predict, y_test)) 
# Trains a K-Nearest Neighbors classifier and evaluates it using a classification report (precision, recall, F1-score).



# Create a Machine Learning Pipeline
pip = Pipeline([('vectorizer', CountVectorizer()),('tfid',TfidfTransformer()), ('knc', KNeighborsClassifier(n_neighbors= 5))])
pip.fit(df['clean_text'], df['label']) 
# Creates a streamlined pipeline that encapsulates vectorization, transformation, and model fitting. This is used for real-time predictions.



# Streamlit Web Interface
st.title('Complaint Classification Portal')
st.write("Enter a consumer complaint. The model will classify it into a complaint category.")

complaint = st.text_area('Enter your Complaint') 

if complaint:
    predict_label = pip.predict([complaint])[0]
    predict_output = label_map[predict_label]
    st.success(f'✅ Predicted Complaint Category: **{predict_output}**')
# Builds an interactive UI where users can enter a complaint. The pipeline predicts and returns the most likely complaint category instantly. 



# Conclusion 
# This project demonstrates a practical application of Natural Language Processing and K-Nearest Neighbors to classify consumer complaints into predefined categories. By preprocessing raw complaint narratives and transforming them into numerical vectors, the model effectively learns patterns across different complaint types.

#Deployed using Streamlit, the app enables real-time classification, making it a valuable tool for customer support teams or regulatory bodies seeking quick categorization of text complaints.

