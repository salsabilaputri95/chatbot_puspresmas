from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re

# Mengunduh stopwords dari nltk
nltk.download('stopwords')

# Inisialisasi aplikasi Flask
app = Flask(__name__)

# Membaca dataset
faq_data = pd.read_excel('dataset.xlsx')

# Preprocessing Teks
stop_words = set(stopwords.words('indonesian'))
stemmer = PorterStemmer()

def preprocess_text(text):
    # Case Folding
    text = text.lower()
    # Tokenization
    words = re.findall(r'\b\w+\b', text)
    # Stopword Removal dan Stemming
    processed_text = [stemmer.stem(word) for word in words if word not in stop_words]
    return ' '.join(processed_text)

# Melakukan preprocessing pada pertanyaan di dataset
faq_data['Processed_Pertanyaan'] = faq_data['Pertanyaan'].apply(preprocess_text)

# TF-IDF Vectorization
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(faq_data['Processed_Pertanyaan'])

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/get_response', methods=['POST'])
def get_response():
    user_query = request.form['user_query']
    
    # Preprocessing query pengguna
    processed_query = preprocess_text(user_query)
    
    # Transform query pengguna ke dalam format TF-IDF
    query_tfidf = vectorizer.transform([processed_query])
    
    # Menghitung cosine similarity
    cosine_similarities = cosine_similarity(query_tfidf, tfidf_matrix)
    
    # Menemukan pertanyaan yang paling mirip
    most_similar_idx = np.argmax(cosine_similarities)
    response = faq_data.iloc[most_similar_idx]['Jawaban']
    
    return response  

if __name__ == '__main__':
    app.run(debug=True)
