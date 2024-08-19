#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from annoy import AnnoyIndex
from joblib import Parallel, delayed
from surprise import Dataset, Reader, SVD, accuracy
from surprise.model_selection import train_test_split as surprise_train_test_split
import joblib
import tensorflow as tf
from sklearn.metrics import mean_squared_error

# Load data
books_df = pd.read_csv('C:\\Users\\sethb\\Documents\\Capstone\\Books.csv', low_memory=False)
ratings_df = pd.read_csv('C:\\Users\\sethb\\Documents\\Capstone\\Ratings.csv')
users_df = pd.read_csv('C:\\Users\\sethb\\Documents\\Capstone\\Users.csv')

# Filter out 0 ratings
ratings_df = ratings_df[ratings_df['Book-Rating'] != 0]

# Clean data
books_df['Book-Author'].fillna('Unknown', inplace=True)
books_df['Publisher'].fillna('Unknown', inplace=True)

users_df['Age'].fillna(users_df['Age'].median(), inplace=True)
median_age = users_df['Age'].median()
users_df.loc[(users_df['Age'] < 5) | (users_df['Age'] > 100), 'Age'] = median_age

# Split location into country only
location_split = users_df['Location'].str.split(',', expand=True)
if location_split.shape[1] > 2:
    users_df['Country'] = location_split[2].str.strip()
else:
    users_df['Country'] = 'Unknown'

# Remove unnecessary columns
books_df.drop(['Image-URL-S', 'Image-URL-M', 'Image-URL-L'], axis=1, inplace=True)
users_df.drop(['Location'], axis=1, inplace=True)

# Set data types
books_df['ISBN'] = books_df['ISBN'].astype(str)
books_df['Year-Of-Publication'] = pd.to_numeric(books_df['Year-Of-Publication'], errors='coerce').fillna(0).astype(int)
books_df['Book-Title'] = books_df['Book-Title'].astype(str)
books_df['Book-Author'] = books_df['Book-Author'].astype(str)
books_df['Publisher'] = books_df['Publisher'].astype(str)

# Filter data by percentiles to reduce size
def filter_data_by_percentiles(df, user_percentile, book_percentile):
    user_counts = df['User-ID'].value_counts()
    book_counts = df['ISBN'].value_counts()
    
    user_threshold = user_counts.quantile(user_percentile)
    book_threshold = book_counts.quantile(book_percentile)
    
    filtered_users = user_counts[user_counts >= user_threshold].index
    filtered_books = book_counts[book_counts >= book_threshold].index
    
    filtered_df = df[df['User-ID'].isin(filtered_users) & df['ISBN'].isin(filtered_books)]
    
    return filtered_df

filtered_ratings_df = filter_data_by_percentiles(ratings_df, 0.9, 0.9)  # Adjust percentiles as needed

# Merge the data
merged_df = filtered_ratings_df.merge(users_df, on='User-ID', how='left')

# Scale age and one-hot encode country
scaler = StandardScaler()
merged_df['Age'] = scaler.fit_transform(merged_df[['Age']])

encoder = OneHotEncoder()
encoded_countries = encoder.fit_transform(merged_df[['Country']])
encoded_countries_df = pd.DataFrame(encoded_countries.toarray(), columns=encoder.get_feature_names_out(['Country']))

# Combine encoded countries with merged_df
merged_df = pd.concat([merged_df.reset_index(drop=True), encoded_countries_df.reset_index(drop=True)], axis=1)
merged_df.drop('Country', axis=1, inplace=True)

# Save the preprocessed data for further steps
preprocessed_data_path = 'C:\\Users\\sethb\\Documents\\Capstone\\preprocessed_data.csv'
processed_books_path = 'C:\\Users\\sethb\\Documents\\Capstone\\processed_books.csv'

merged_df.to_csv(preprocessed_data_path, index=False)
books_df.to_csv(processed_books_path, index=False)

print(f"Preprocessed data saved to: {preprocessed_data_path}")
print(f"Processed books data saved to: {processed_books_path}")


# In[2]:


# Load preprocessed data
ratings_df = pd.read_csv(preprocessed_data_path)
books_df = pd.read_csv(processed_books_path)

# Prepare data for surprise
reader = Reader(rating_scale=(1, 10))
data = Dataset.load_from_df(ratings_df[['User-ID', 'ISBN', 'Book-Rating']], reader)

# Train-test split
trainset, testset = surprise_train_test_split(data, test_size=0.25, random_state=42)

# Train SVD model
algo = SVD()
algo.fit(trainset)
predictions = algo.test(testset)

# Calculate RMSE for the collaborative filtering model
cf_rmse = accuracy.rmse(predictions, verbose=True)
print(f"Collaborative filtering model RMSE: {cf_rmse}")

# Save the trained SVD model
joblib.dump(algo, 'C:\\Users\\sethb\\Documents\\Capstone\\svd_model.pkl')

