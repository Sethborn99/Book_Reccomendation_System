#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.neural_network import MLPRegressor

# Load data function
def load_data():
    books_df = pd.read_csv('C:\\Users\\sethb\\Documents\\Capstone\\Books.csv')
    ratings_df = pd.read_csv('C:\\Users\\sethb\\Documents\\Capstone\\Ratings.csv')
    users_df = pd.read_csv('C:\\Users\\sethb\\Documents\\Capstone\\Users.csv')
    return books_df, ratings_df, users_df

# Preprocess data
def preprocess_data(all_ratings, books_df, new_user_id, books_to_rate, age, location):
    reader = Reader(rating_scale=(1, 10))
    data = Dataset.load_from_df(all_ratings[['User-ID', 'ISBN', 'Book-Rating']], reader)

    trainset = data.build_full_trainset()
    algo = SVD()
    algo.fit(trainset)

    all_books = set(books_df['ISBN'])
    rated_books = set(books_to_rate.keys())
    unrated_books = all_books - rated_books

    predictions = [algo.predict(new_user_id, book) for book in unrated_books]
    predictions.sort(key=lambda x: x.est, reverse=True)

    pred_df = pd.DataFrame(predictions, columns=['User-ID', 'ISBN', 'Book-Rating', 'Est', 'Details'])
    pred_df = pred_df[['ISBN', 'Est']]
    pred_df = pred_df.merge(books_df, on='ISBN', how='left')
    pred_df['User-ID'] = new_user_id
    pred_df['Age'] = age
    pred_df['Location'] = location

    # Add actual ratings back into pred_df
    rated_books_df = pd.DataFrame(list(books_to_rate.items()), columns=['ISBN', 'Book-Rating'])
    rated_books_df['User-ID'] = new_user_id
    pred_df = pd.concat([pred_df, rated_books_df])

    return pred_df

# Collaborative Filtering Model
def collaborative_filtering(ratings_df):
    reader = Reader(rating_scale=(1, 10))
    data = Dataset.load_from_df(ratings_df[['User-ID', 'ISBN', 'Book-Rating']], reader)

    trainset, testset = train_test_split(data, test_size=0.25)

    algo = SVD()
    algo.fit(trainset)

    predictions = algo.test(testset)

    return algo, predictions

# Content-Based Filtering Model
def content_based_filtering(books_df):
    # Implement your content-based filtering logic here
    pass

# Deep Learning Model
def deep_learning_model():
    # Implement your deep learning model logic here
    pass

# Hybrid Model
def hybrid_model(all_ratings, books_df, users_df, new_user_id, books_to_rate, age, location):
    pred_df = preprocess_data(all_ratings, books_df, new_user_id, books_to_rate, age, location)

    # One-hot encode location and scale age
    ct = ColumnTransformer([
        ('ohe', OneHotEncoder(), ['Location']),
        ('scaler', StandardScaler(), ['Age'])
    ], remainder='passthrough')

    X = pred_df[['Est', 'Age', 'Location']]
    X_transformed = ct.fit_transform(X)

    # Prepare target variable
    y = pred_df['Book-Rating']

    # Train the linear regression model
    lin_reg = LinearRegression()
    lin_reg.fit(X_transformed, y)

    # Predict using the linear regression model
    y_pred = lin_reg.predict(X_transformed)
    pred_df['Predicted_Rating'] = y_pred

    # Get the top 10 book recommendations based on linear regression predictions
    top_recommendations = pred_df.sort_values(by='Predicted_Rating', ascending=False).head(10)

    return top_recommendations


# In[ ]:




