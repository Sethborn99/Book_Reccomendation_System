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


# In[3]:


# Filter books to reduce size for testing and remove sparse data
min_ratings = 10  # Minimum number of ratings to include book
books_with_enough_ratings = ratings_df['ISBN'].value_counts()
books_with_enough_ratings = books_with_enough_ratings[books_with_enough_ratings >= min_ratings].index
books_subset = books_df[books_df['ISBN'].isin(books_with_enough_ratings)]

ratings_subset = ratings_df[ratings_df['ISBN'].isin(books_subset['ISBN'])]

# Combine title, author, and publisher into a single feature for content-based filtering
books_subset['metadata'] = books_subset['Book-Title'] + ' ' + books_subset['Book-Author'] + ' ' + books_subset['Publisher']

# Use TF-IDF Vectorizer to transform metadata into a matrix of TF-IDF features
tfidf = TfidfVectorizer(stop_words='english', max_features=5000, min_df=2, max_df=0.8)
tfidf_matrix = tfidf.fit_transform(books_subset['metadata'])

# Reduce dimensionality of the TF-IDF matrix using Truncated SVD
n_components = 50
svd = TruncatedSVD(n_components=n_components)
tfidf_matrix_reduced = svd.fit_transform(tfidf_matrix)

# Create an Annoy index for approximate nearest neighbors search
f = tfidf_matrix_reduced.shape[1]
annoy_index = AnnoyIndex(f, 'angular')

def add_item_to_annoy(index, vector):
    annoy_index.add_item(index, vector)

# Add items to the Annoy index
for i in range(tfidf_matrix_reduced.shape[0]):
    add_item_to_annoy(i, tfidf_matrix_reduced[i])

annoy_index.build(10)  # You can increase the number of trees for better accuracy

# Create a dictionary for quick lookup of book ratings
ratings_dict = ratings_subset.groupby('ISBN')['Book-Rating'].mean().to_dict()

# Cache content-based scores to minimize Annoy index lookups
content_based_scores = {}

def get_content_based_score(isbn, annoy_index=annoy_index, n=10):
    if isbn in content_based_scores:
        return content_based_scores[isbn]
    if isbn not in books_subset['ISBN'].values:
        return []  # Return an empty list if the ISBN is not found
    idx = books_subset.index[books_subset['ISBN'] == isbn].tolist()[0]
    if idx >= annoy_index.get_n_items():
        return []  # Skip if index is out of bounds
    similar_indices = annoy_index.get_nns_by_item(idx, n + 1)[1:]  # Get n similar items, exclude the item itself
    similarities = [1 - annoy_index.get_distance(idx, i) for i in similar_indices]
    similarities = np.clip(similarities, 0, 1)  # Clip similarities to range [0, 1]
    similar_isbns = books_subset.iloc[similar_indices]['ISBN'].tolist()
    scores = list(zip(similar_isbns, similarities))
    content_based_scores[isbn] = scores
    return scores

# Function to calculate RMSE for content-based filtering
def calculate_cb_rmse(ratings_df, annoy_index, n=10):
    true_ratings = []
    predicted_ratings = []
    for isbn, rating in zip(ratings_df['ISBN'], ratings_df['Book-Rating']):
        cb_scores = get_content_based_score(isbn, annoy_index, n)
        if cb_scores:
            # Debug: Print the content-based scores
            print(f"Content-based scores for ISBN {isbn}:", cb_scores)
            filtered_scores = [(similar_isbn, similarity) for similar_isbn, similarity in cb_scores if similarity > 0]
            weighted_sum = sum(similarity for _, similarity in filtered_scores)
            if weighted_sum == 0:
                cb_rating = 0
            else:
                weighted_ratings_sum = sum(ratings_dict.get(similar_isbn, 0) * similarity for similar_isbn, similarity in filtered_scores)
                cb_rating = weighted_ratings_sum / weighted_sum
            predicted_ratings.append(cb_rating)
            true_ratings.append(rating)
    # Ensure no NaN values in predictions
    predicted_ratings = [pred if not np.isnan(pred) else 0 for pred in predicted_ratings]
    return mean_squared_error(true_ratings, predicted_ratings, squared=False)

# Calculate RMSE for content-based filtering
cb_rmse = calculate_cb_rmse(ratings_subset, annoy_index)
print(f"Content-based filtering model RMSE: {cb_rmse}")


# In[4]:


def generate_features(user_id, isbn, algo, annoy_index, books_subset, ratings_dict, n=10):
    # Collaborative filtering score
    cf_score = algo.predict(user_id, isbn).est
    
    # Content-based score
    if isbn in books_subset['ISBN'].values:
        idx = books_subset.index[books_subset['ISBN'] == isbn].tolist()[0]
        if idx >= annoy_index.get_n_items():
            cb_score = 0
        else:
            similar_indices = annoy_index.get_nns_by_item(idx, n + 1)[1:]
            similarities = [1 - annoy_index.get_distance(idx, i) for i in similar_indices]
            similarities = np.clip(similarities, 0, 1)
            similar_isbns = books_subset.iloc[similar_indices]['ISBN'].tolist()
            scores = list(zip(similar_isbns, similarities))
            weighted_sum = sum(similarity for _, similarity in scores)
            if weighted_sum == 0:
                cb_score = 0
            else:
                weighted_ratings_sum = sum(ratings_dict.get(similar_isbn, 0) * similarity for similar_isbn, similarity in scores)
                cb_score = weighted_ratings_sum / weighted_sum
    else:
        cb_score = 0

    return [cf_score, cb_score]

# Generate training data
X_train = []
y_train = []

for user_id, isbn, rating in zip(ratings_subset['User-ID'], ratings_subset['ISBN'], ratings_subset['Book-Rating']):
    features = generate_features(user_id, isbn, algo, annoy_index, books_subset, ratings_dict)
    X_train.append(features)
    y_train.append(rating)

X_train = np.array(X_train, dtype=np.float32)
y_train = np.array(y_train, dtype=np.float32)


# In[5]:


# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Define the meta-level learner
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(64, input_dim=X_train.shape[1], activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(1, activation='linear')
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Train the meta-level model
history = model.fit(X_train, y_train, epochs=50, batch_size=256, validation_data=(X_val, y_val),
                    callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)])

# Save the model
model.save('C:\\Users\\sethb\\Documents\\Capstone\\meta_model.h5')

# Load the trained model
meta_model = tf.keras.models.load_model('C:\\Users\\sethb\\Documents\\Capstone\\meta_model.h5')


# In[6]:


# Evaluate RMSE for Collaborative Filtering
y_true_cf = []
y_pred_cf = []
for user_id, isbn, rating in zip(ratings_subset['User-ID'], ratings_subset['ISBN'], ratings_subset['Book-Rating']):
    y_true_cf.append(rating)
    y_pred_cf.append(algo.predict(user_id, isbn).est)
rmse_cf = mean_squared_error(y_true_cf, y_pred_cf, squared=False)
print(f"Collaborative Filtering RMSE: {rmse_cf}")

# Evaluate RMSE for Content-Based Filtering
y_true_cb = []
y_pred_cb = []
for isbn, rating in zip(ratings_subset['ISBN'], ratings_subset['Book-Rating']):
    y_true_cb.append(rating)
    cb_scores = get_content_based_score(isbn, annoy_index, n=10)
    if cb_scores:
        filtered_scores = [(similar_isbn, similarity) for similar_isbn, similarity in cb_scores if similarity > 0]
        weighted_sum = sum(similarity for _, similarity in filtered_scores)
        if weighted_sum == 0:
            cb_rating = 0
        else:
            weighted_ratings_sum = sum(ratings_dict.get(similar_isbn, 0) * similarity for similar_isbn, similarity in filtered_scores)
            cb_rating = weighted_ratings_sum / weighted_sum
        y_pred_cb.append(cb_rating)
    else:
        y_pred_cb.append(0)
rmse_cb = mean_squared_error(y_true_cb, y_pred_cb, squared=False)
print(f"Content-Based Filtering RMSE: {rmse_cb}")

# Evaluate RMSE for Meta-Level Model
X_val_meta = []
y_val_meta = []
for user_id, isbn, rating in zip(ratings_subset['User-ID'], ratings_subset['ISBN'], ratings_subset['Book-Rating']):
    features = generate_features(user_id, isbn, algo, annoy_index, books_subset, ratings_dict)
    X_val_meta.append(features)
    y_val_meta.append(rating)

X_val_meta = np.array(X_val_meta, dtype=np.float32)
y_val_meta = np.array(y_val_meta, dtype=np.float32)

y_pred_meta = meta_model.predict(X_val_meta)
rmse_meta = mean_squared_error(y_val_meta, y_pred_meta, squared=False)
print(f"Meta-Level Model RMSE: {rmse_meta}")


# In[ ]:




