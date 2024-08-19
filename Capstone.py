#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install --upgrade numpy --user')
get_ipython().system('pip install --upgrade scipy scikit-learn --user')


# In[1]:


# Import necessary libraries

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


# In[12]:


# Define the data types for each column
books_dtypes = {
    'ISBN': str,
    'Book-Title': str,
    'Book-Author': str,
    'Year-Of-Publication': str,  # Load as string initially
    'Publisher': str,
    'Image-URL-S': str,
    'Image-URL-M': str,
    'Image-URL-L': str
}

ratings_dtypes = {
    'User-ID': str, 
    'ISBN': str,
    'Book-Rating': int
}

users_dtypes = {
    'User-ID': str,  
    'Location': str,
    'Age': float  # Use float to handle any missing values in Age
}


# In[13]:


# Load the datasets with specified data types
books = pd.read_csv('C:\\Users\\sethb\\Documents\\Capstone\\Books.csv', dtype=books_dtypes)
ratings = pd.read_csv('C:\\Users\\sethb\\Documents\\Capstone\\Ratings.csv', dtype=ratings_dtypes)
users = pd.read_csv('C:\\Users\\sethb\\Documents\\Capstone\\Users.csv', dtype=users_dtypes)


# In[15]:


# Display the first few rows of each dataset
print("Books Data:")
print(books.head())
print("\nRatings Data:")
print(ratings.head())
print("\nUsers Data:")
print(users.head())


# In[16]:


# Convert Year-Of-Publication to numeric, setting errors='coerce' to handle non-numeric values
books['Year-Of-Publication'] = pd.to_numeric(books['Year-Of-Publication'], errors='coerce')


# In[17]:


print("Books Data Types:")
print(books.dtypes)
print("\nRatings Data Types:")
print(ratings.dtypes)
print("\nUsers Data Types:")
print(users.dtypes)


# In[18]:


# Data Cleaning and Preprocessing

# Check for missing values
print("Missing values in Books dataset:")
print(books.isnull().sum())
print("\nMissing values in Ratings dataset:")
print(ratings.isnull().sum())
print("\nMissing values in Users dataset:")
print(users.isnull().sum())


# In[19]:


# Handle missing values in Books dataset
books['Book-Author'].fillna('Unknown', inplace=True)
books['Publisher'].fillna('Unknown', inplace=True)
books['Image-URL-S'].fillna('No URL', inplace=True)
books['Image-URL-M'].fillna('No URL', inplace=True)
books['Image-URL-L'].fillna('No URL', inplace=True)
books['Year-Of-Publication'].fillna(0, inplace=True)  # Fill with 0 or a placeholder value


# In[20]:


# Handle missing values in Users dataset by filling with the median age
users['Age'].fillna(users['Age'].median(), inplace=True)


# In[21]:


# Normalize data (e.g., scaling ratings)
scaler = MinMaxScaler()
ratings['Book-Rating'] = scaler.fit_transform(ratings[['Book-Rating']])


# In[22]:


# Preprocess text data (e.g., book titles, authors)
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()


# In[23]:


# Preprocess text data (e.g., book titles, authors)
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def preprocess_text(text):
    # Lowercase
    text = text.lower()
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    # Remove stopwords and stem
    text = ' '.join([stemmer.stem(word) for word in text.split() if word not in stop_words])
    return text

books['Book-Title'] = books['Book-Title'].apply(preprocess_text)
books['Book-Author'] = books['Book-Author'].apply(preprocess_text)

print("Processed Books Data:")
print(books.head())


# In[24]:


# Verify data after handling missing values
print("Books Data after Handling Missing Values:")
print(books.head())
print("\nRatings Data after Handling Missing Values:")
print(ratings.head())
print("\nUsers Data after Handling Missing Values:")
print(users.head())


# In[25]:


# Check for any remaining missing values
print("Missing values in Books dataset after Handling Missing Values:")
print(books.isnull().sum())
print("\nMissing values in Ratings dataset after Handling Missing Values:")
print(ratings.isnull().sum())
print("\nMissing values in Users dataset after Handling Missing Values:")
print(users.isnull().sum())


# In[26]:


# Building the Recommender System

from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Collaborative Filtering (User-User)
def collaborative_filtering(user_id, top_n=10):
    user_ratings = ratings.pivot(index='User-ID', columns='ISBN', values='Rating')
    user_similarity = cosine_similarity(user_ratings.fillna(0))
    user_sim_df = pd.DataFrame(user_similarity, index=user_ratings.index, columns=user_ratings.index)
    
    similar_users = user_sim_df[user_id].sort_values(ascending=False)[1:top_n+1]
    similar_users_ratings = user_ratings.loc[similar_users.index]
    return similar_users_ratings.mean().sort_values(ascending=False).head(top_n)

# Content-Based Filtering
def content_based_filtering(book_isbn, top_n=10):
    book_features = books[['Book-Title', 'Book-Author']]
    book_features = book_features.set_index('Book-Title')
    similarity_matrix = cosine_similarity(book_features, book_features)
    book_sim_df = pd.DataFrame(similarity_matrix, index=book_features.index, columns=book_features.index)
    
    similar_books = book_sim_df.loc[book_isbn].sort_values(ascending=False)[1:top_n+1]
    return similar_books

# Deep Learning Model (Neural Collaborative Filtering)
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Flatten, Dot, Add, Dense

num_users = len(ratings['User-ID'].unique())
num_books = len(ratings['ISBN'].unique())

user_input = Input(shape=(1,))
user_embedding = Embedding(num_users, 50)(user_input)
user_vec = Flatten()(user_embedding)

book_input = Input(shape=(1,))
book_embedding = Embedding(num_books, 50)(book_input)
book_vec = Flatten()(book_embedding)

dot_product = Dot(axes=1)([user_vec, book_vec])
output = Dense(1)(dot_product)

model = Model([user_input, book_input], output)
model.compile(optimizer='adam', loss='mean_squared_error')

user_ids = ratings['User-ID'].astype('category').cat.codes.values
book_ids = ratings['ISBN'].astype('category').cat.codes.values

model.fit([user_ids, book_ids], ratings['Rating'], epochs=5, verbose=1)

def deep_learning_recommender(user_id, top_n=10):
    user_index = np.where(user_ids == user_id)[0][0]
    book_indices = np.arange(num_books)
    predictions = model.predict([np.full(num_books, user_index), book_indices])
    recommended_books = np.argsort(predictions[:, 0])[-top_n:]
    return books.iloc[recommended_books]

# Hybrid Recommender System
def hybrid_recommender(user_id, book_isbn, top_n=10):
    cf_recommendations = collaborative_filtering(user_id, top_n)
    cb_recommendations = content_based_filtering(book_isbn, top_n)
    dl_recommendations = deep_learning_recommender(user_id, top_n)
    
    combined_recommendations = cf_recommendations.index.union(cb_recommendations.index).union(dl_recommendations.index)
    return combined_recommendations[:top_n]

# Example usage
user_id_example = 1
book_isbn_example = '034545104X'
print("Hybrid Recommendations for User-ID 1 and ISBN '034545104X':")
print(hybrid_recommender(user_id_example, book_isbn_example))


# In[ ]:




