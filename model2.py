#!/usr/bin/env python
# coding: utf-8

# In[27]:


import pandas as pd
from surprise import Dataset, Reader, SVD, accuracy
from surprise.model_selection import train_test_split as surprise_train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from annoy import AnnoyIndex
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np
from joblib import Parallel, delayed

# Load preprocessed data
preprocessed_data_path = 'C:\\Users\\sethb\\Documents\\Capstone\\preprocessed_data.csv'
processed_books_path = 'C:\\Users\\sethb\\Documents\\Capstone\\processed_books.csv'

merged_df = pd.read_csv(preprocessed_data_path)
books_df = pd.read_csv(processed_books_path)


# In[22]:


# Step 1: Build and Evaluate Collaborative Filtering Model

# Prepare data for Surprise
reader = Reader(rating_scale=(1, 10))
data = Dataset.load_from_df(merged_df[['User-ID', 'ISBN', 'Book-Rating']], reader)

# Split the data into training and test sets
trainset, testset = surprise_train_test_split(data, test_size=0.2)

# Train the SVD model
collab_svd = SVD()
collab_svd.fit(trainset)

# Evaluate the model
predictions = collab_svd.test(testset)
rmse = accuracy.rmse(predictions)
print(f'Collaborative Filtering Model RMSE: {rmse}')


# In[23]:


# Step 2: Build and Evaluate Content-Based Filtering Model

# Combine book features
books_df['Combined-Features'] = books_df['Book-Title'] + ' ' + books_df['Book-Author'] + ' ' + books_df['Publisher']

# Apply TF-IDF
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(books_df['Combined-Features'])

# Apply Truncated SVD for dimensionality reduction
svd = TruncatedSVD(n_components=50)
svd_matrix = svd.fit_transform(tfidf_matrix)

# Create Annoy index
annoy_index = AnnoyIndex(50, 'angular')
for i in range(len(svd_matrix)):
    annoy_index.add_item(i, svd_matrix[i])
annoy_index.build(10)

# Function to get content-based predictions
def get_content_predictions(user_id, book_id):
    book_rows = books_df[books_df['ISBN'] == book_id]
    if book_rows.empty:
        return merged_df['Book-Rating'].mean()  # Return mean rating if book is not found
    book_index = book_rows.index[0]
    similar_books = annoy_index.get_nns_by_item(book_index, 10)
    similar_ratings = merged_df[(merged_df['User-ID'] == user_id) & (merged_df['ISBN'].isin(books_df.iloc[similar_books]['ISBN']))]['Book-Rating']
    if similar_ratings.empty:
        return merged_df['Book-Rating'].mean()  # Return mean rating if no similar books are rated
    return similar_ratings.mean()


# In[35]:


# Step 3: Create and Evaluate Meta-Level Model with TensorFlow/Keras
print("Creating and evaluating meta-level model...")

# Generate predictions from collaborative filtering model
collab_predictions = [collab_svd.predict(row['User-ID'], row['ISBN']).est for _, row in merged_df.iterrows()]
print("Collaborative filtering predictions generated.")

# Generate predictions from content-based filtering model using parallel processing
print("Generating content-based filtering predictions...")

def generate_content_predictions_parallel():
    results = []
    for idx, row in enumerate(merged_df.itertuples(index=False), 1):
        if idx % 100 == 0:  # Print progress every 100 rows
            print(f"Processed {idx} rows... User-ID: {row._1}, ISBN: {row._2}")
        results.append(get_content_predictions(row._1, row._2))  # Access attributes by actual column names
    return results

# Correct column names used by itertuples
def generate_content_predictions_parallel():
    results = []
    for idx, row in enumerate(merged_df.itertuples(index=False, name=None), 1):
        if idx % 100 == 0:  # Print progress every 100 rows
            print(f"Processed {idx} rows... User-ID: {row[0]}, ISBN: {row[1]}")
        results.append(get_content_predictions(row[0], row[1]))  # Access attributes by actual column names
    return results

content_predictions = generate_content_predictions_parallel()
print("Content-based filtering predictions generated.")

# Combine the predictions as features for the meta-level model
meta_input = np.vstack([collab_predictions, content_predictions]).T
meta_output = merged_df['Book-Rating'].values

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(meta_input, meta_output, test_size=0.2, random_state=42)
print("Data split into training and validation sets.")


# In[36]:


import tensorflow as tf

# Define the meta-level learner
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(64, input_dim=X_train.shape[1], activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(1, activation='linear')
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])
print("Meta-level learner defined and compiled.")

# Train the meta-level model with early stopping
history = model.fit(X_train, y_train, epochs=50, batch_size=256, validation_data=(X_val, y_val),
                    callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)])
print("Meta-level model trained.")

# Save the model
model_path = 'C:\\Users\\sethb\\Documents\\Capstone\\meta_model.h5'
model.save(model_path)
print(f"Model saved to {model_path}.")

# Load the trained model
meta_model = tf.keras.models.load_model(model_path)
print("Trained model loaded.")

# Evaluate RMSE for Meta-Level Model
y_pred_meta = meta_model.predict(X_val)
rmse_meta = mean_squared_error(y_val, y_pred_meta, squared=False)
print(f"Meta-Level Model RMSE: {rmse_meta}")


# In[37]:


model.save('C:\\Users\\sethb\\Documents\\Capstone\\meta_model.keras')


# In[38]:


import matplotlib.pyplot as plt

# Visualization: Training and Validation Loss
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Visualization: Predicted vs Actual Ratings
plt.figure(figsize=(10, 6))
plt.scatter(y_val, y_pred_meta, alpha=0.3)
plt.title('Predicted vs Actual Ratings')
plt.xlabel('Actual Ratings')
plt.ylabel('Predicted Ratings')
plt.plot([min(y_val), max(y_val)], [min(y_val), max(y_val)], 'r')
plt.show()


# In[40]:


import numpy as np
import matplotlib.pyplot as plt

# Assuming y_val and y_pred_meta are already defined

# Compute counts for each rating for true and predicted ratings
bins = np.arange(1, 12) - 0.5
true_counts, _ = np.histogram(y_val, bins=bins)
pred_counts, _ = np.histogram(y_pred_meta, bins=bins)
bin_centers = 0.5 * (bins[1:] + bins[:-1])

# Plot the bar chart
plt.figure(figsize=(12, 6))
bar_width = 0.4

plt.bar(bin_centers - bar_width / 2, true_counts, width=bar_width, color='blue', label='True Ratings')
plt.bar(bin_centers + bar_width / 2, pred_counts, width=bar_width, color='red', label='Predicted Ratings')

plt.xlabel('Rating')
plt.ylabel('Count')
plt.title('Distribution of True Ratings and Predicted Ratings')
plt.xticks(np.arange(1, 11))
plt.legend()
plt.show()


# In[46]:


import numpy as np
import matplotlib.pyplot as plt

# Assuming y_val, y_pred_meta, collab_predictions, and content_predictions are already defined

# Ensure predictions are for the same validation set
collab_predictions_val = collab_predictions[:len(y_val)]
content_predictions_val = content_predictions[:len(y_val)]

# Calculate RMSE for each model
rmse_collab = mean_squared_error(y_val, collab_predictions_val, squared=False)
rmse_content = mean_squared_error(y_val, content_predictions_val, squared=False)
rmse_meta = mean_squared_error(y_val, y_pred_meta, squared=False)

# Bin settings for histogram
bins = np.arange(1, 12) - 0.5
bin_centers = 0.5 * (bins[1:] + bins[:-1])
bar_width = 0.4

# True Ratings
true_counts, _ = np.histogram(y_val, bins=bins)

# Collaborative Filtering Model
collab_counts, _ = np.histogram(collab_predictions_val, bins=bins)

# Content-Based Filtering Model
content_counts, _ = np.histogram(content_predictions_val, bins=bins)

# Meta-Level Model (from previous predictions)
meta_counts, _ = np.histogram(y_pred_meta, bins=bins)

# Create subplots
fig, axs = plt.subplots(3, 1, figsize=(12, 18))

# Collaborative Filtering Model
axs[0].bar(bin_centers - bar_width / 2, true_counts, width=bar_width, color='blue', label='True Ratings')
axs[0].bar(bin_centers + bar_width / 2, collab_counts, width=bar_width, color='red', label='Collaborative Predictions')
axs[0].set_xlabel('Rating')
axs[0].set_ylabel('Count')
axs[0].set_title(f'Distribution of True Ratings and Collaborative Predictions (RMSE: {rmse_collab:.4f})')
axs[0].set_xticks(np.arange(1, 11))
axs[0].legend()

# Content-Based Filtering Model
axs[1].bar(bin_centers - bar_width / 2, true_counts, width=bar_width, color='blue', label='True Ratings')
axs[1].bar(bin_centers + bar_width / 2, content_counts, width=bar_width, color='green', label='Content-Based Predictions')
axs[1].set_xlabel('Rating')
axs[1].set_ylabel('Count')
axs[1].set_title(f'Distribution of True Ratings and Content-Based Predictions (RMSE: {rmse_content:.4f})')
axs[1].set_xticks(np.arange(1, 11))
axs[1].legend()

# Meta-Level Model
axs[2].bar(bin_centers - bar_width / 2, true_counts, width=bar_width, color='blue', label='True Ratings')
axs[2].bar(bin_centers + bar_width / 2, meta_counts, width=bar_width, color='purple', label='Meta-Level Predictions')
axs[2].set_xlabel('Rating')
axs[2].set_ylabel('Count')
axs[2].set_title(f'Distribution of True Ratings and Meta-Level Predictions (RMSE: {rmse_meta:.4f})')
axs[2].set_xticks(np.arange(1, 11))
axs[2].legend()

plt.tight_layout()
plt.show()


# In[51]:


from surprise import accuracy
from sklearn.metrics import mean_absolute_error
import numpy as np
from collections import defaultdict

# Example of calculating MAE
mae = accuracy.mae(predictions)
print(f'Collaborative Filtering Model MAE: {mae}')

# Example of calculating Precision@k and Recall@k
def precision_recall_at_k(predictions, k=10, threshold=7):
    user_est_true = defaultdict(list)
    for uid, _, true_r, est, _ in predictions:
        user_est_true[uid].append((est, true_r))

    precisions = dict()
    recalls = dict()

    for uid, user_ratings in user_est_true.items():
        user_ratings.sort(key=lambda x: x[0], reverse=True)
        n_rel = sum((true_r >= threshold) for (_, true_r) in user_ratings)
        n_rec_k = sum((est >= threshold) for (est, _) in user_ratings[:k])
        n_rel_and_rec_k = sum(((true_r >= threshold) and (est >= threshold))
                              for (est, true_r) in user_ratings[:k])

        precisions[uid] = n_rel_and_rec_k / n_rec_k if n_rec_k != 0 else 1
        recalls[uid] = n_rel_and_rec_k / n_rel if n_rel != 0 else 1

    precision_at_k = sum(prec for prec in precisions.values()) / len(precisions)
    recall_at_k = sum(rec for rec in recalls.values()) / len(recalls)
    
    return precision_at_k, recall_at_k

precision_at_k, recall_at_k = precision_recall_at_k(predictions, k=10)
print(f'Precision@10: {precision_at_k}')
print(f'Recall@10: {recall_at_k}')

# Example of calculating NDCG
def ndcg_at_k(predictions, k=10):
    user_est_true = defaultdict(list)
    for uid, _, true_r, est, _ in predictions:
        user_est_true[uid].append((est, true_r))

    def dcg(relevance_scores):
        return sum([rel / np.log2(idx + 2) for idx, rel in enumerate(relevance_scores)])

    ndcgs = []

    for uid, user_ratings in user_est_true.items():
        user_ratings.sort(key=lambda x: x[0], reverse=True)
        relevance_scores = [true_r for (_, true_r) in user_ratings[:k]]
        idcg = dcg(sorted(relevance_scores, reverse=True))
        dcg_score = dcg(relevance_scores)
        ndcg_score = dcg_score / idcg if idcg != 0 else 0
        ndcgs.append(ndcg_score)
    
    return np.mean(ndcgs)

ndcg = ndcg_at_k(predictions, k=10)
print(f'NDCG@10: {ndcg}')


# In[ ]:




