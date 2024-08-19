#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!/usr/bin/env python
# coding: utf-8

import pandas as pd
from surprise import Dataset, Reader, SVD, accuracy
from surprise.model_selection import train_test_split as surprise_train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from annoy import AnnoyIndex
from sklearn.metrics import mean_squared_error
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import matplotlib.pyplot as plt

# Load preprocessed data
preprocessed_data_path = 'C:\\Users\\sethb\\Documents\\Capstone\\preprocessed_data.csv'
processed_books_path = 'C:\\Users\\sethb\\Documents\\Capstone\\processed_books.csv'

merged_df = pd.read_csv(preprocessed_data_path)
books_df = pd.read_csv(processed_books_path)

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
collab_rmse = accuracy.rmse(predictions)
print(f'Collaborative Filtering Model RMSE: {collab_rmse}')

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
annoy_index.build(10)  # 10 trees

# Step 3: Generate Content Predictions in Parallel

def get_content_predictions(user_id, isbn):
    book_idx = books_df.index[books_df['ISBN'] == isbn].tolist()
    if not book_idx:
        return 5.0  # Default rating if the book is not found
    similar_items = annoy_index.get_nns_by_item(book_idx[0], 10)
    similar_ratings = merged_df[(merged_df['ISBN'].isin(books_df.iloc[similar_items]['ISBN'])) & (merged_df['User-ID'] == user_id)]['Book-Rating']
    if similar_ratings.empty:
        return 5.0  # Default rating if no similar ratings are found
    return np.clip(similar_ratings.mean(), 1, 10)  # Ensure rating is between 1 and 10

def generate_content_predictions_parallel():
    results = []
    with ThreadPoolExecutor() as executor:
        future_to_row = {
            executor.submit(get_content_predictions, row[0], row[1]): row
            for row in merged_df.itertuples(index=False, name=None)
        }
        for idx, future in enumerate(future_to_row, 1):
            row = future_to_row[future]
            if idx % 100 == 0:  # Print progress every 100 rows
                print(f"Processed {idx} rows... User-ID: {row[0]}, ISBN: {row[1]}")
            results.append(future.result())
    return results

content_predictions = generate_content_predictions_parallel()
print(f"Generated {len(content_predictions)} content predictions.")

# Evaluate content-based model
content_rmse = mean_squared_error(merged_df['Book-Rating'], content_predictions, squared=False)
print(f'Content-Based Filtering Model RMSE: {content_rmse}')

# Step 4: Meta-Level Model

# Prepare data for meta-level model
collab_predictions = [collab_svd.predict(row['User-ID'], row['ISBN']).est for idx, row in merged_df.iterrows()]

X = np.column_stack((collab_predictions, content_predictions))
y = merged_df['Book-Rating'].values

# Split the data into training and validation sets
train_size = int(0.8 * len(X))
X_train, X_val = X[:train_size], X[train_size:]
y_train, y_val = y[:train_size], y[train_size:]

# Build the meta-level model
meta_model = Sequential([
    Dense(128, input_dim=X_train.shape[1], activation='relu'),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='linear')
])

meta_model.compile(optimizer='adam', loss='mean_squared_error')

# Train the meta-level model
history = meta_model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_val, y_val))

# Evaluate the meta-level model
y_pred_meta = meta_model.predict(X_val)
meta_rmse = mean_squared_error(y_val, y_pred_meta, squared=False)
print(f"Meta-Level Model RMSE: {meta_rmse}")

# Save the trained model
model_path = 'C:\\Users\\sethb\\Documents\\Capstone\\meta_model.keras'
meta_model.save(model_path)
print(f"Model saved to {model_path}.")

# Load the trained model (for later use)
meta_model = tf.keras.models.load_model(model_path)
print("Trained model loaded.")

# Visualization: Training and Validation Loss
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Visualization: Predicted vs Actual Ratings for each model
def plot_pred_vs_actual(y_true, y_pred, model_name, rmse):
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred, alpha=0.3)
    plt.title(f'Predicted vs Actual Ratings ({model_name}) - RMSE: {rmse:.4f}')
    plt.xlabel('Actual Ratings')
    plt.ylabel('Predicted Ratings')
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'r')
    plt.show()

plot_pred_vs_actual(merged_df['Book-Rating'], collab_predictions, 'Collaborative Filtering', collab_rmse)
plot_pred_vs_actual(merged_df['Book-Rating'], content_predictions, 'Content-Based Filtering', content_rmse)
plot_pred_vs_actual(y_val, y_pred_meta, 'Meta-Level Model', meta_rmse)

# Compute counts for each rating for true and predicted ratings
def plot_rating_distribution(y_true, y_pred, model_name, rmse):
    bins = np.arange(1, 12) - 0.5
    true_counts, _ = np.histogram(y_true, bins=bins)
    pred_counts, _ = np.histogram(y_pred, bins=bins)
    bin_centers = 0.5 * (bins[1:] + bins[:-1])

    plt.figure(figsize=(12, 6))
    bar_width = 0.4

    plt.bar(bin_centers - bar_width / 2, true_counts, width=bar_width, color='blue', label='True Ratings')
    plt.bar(bin_centers + bar_width / 2, pred_counts, width=bar_width, color='red', label='Predicted Ratings')

    plt.xlabel('Rating')
    plt.ylabel('Count')
    plt.title(f'Distribution of True Ratings and Predicted Ratings ({model_name}) - RMSE: {rmse:.4f}')
    plt.xticks(np.arange(1, 11))
    plt.legend()
    plt.show()

plot_rating_distribution(merged_df['Book-Rating'], collab_predictions, 'Collaborative Filtering', collab_rmse)
plot_rating_distribution(merged_df['Book-Rating'], content_predictions, 'Content-Based Filtering', content_rmse)
plot_rating_distribution(y_val, y_pred_meta, 'Meta-Level Model', meta_rmse)


# In[2]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns



# Display the first few rows of each dataset
print("Merged DataFrame:")
print(merged_df.head())

print("\nBooks DataFrame:")
print(books_df.head())


# In[3]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Visualize the distribution of ratings with counts
plt.figure(figsize=(10, 6))
ax = sns.histplot(merged_df['Book-Rating'], bins=10)
plt.title('Distribution of Book Ratings')
plt.xlabel('Rating')
plt.ylabel('Count')

# Add counts on top of the bars
for p in ax.patches:
    ax.annotate(f'\n{int(p.get_height())}', (p.get_x() + p.get_width() / 2, p.get_height()), 
                ha='center', va='center', fontsize=12, color='black', xytext=(0, 10), 
                textcoords='offset points')
plt.show()

# Examine the number of ratings per user with log scale on x-axis
ratings_per_user = merged_df.groupby('User-ID').size()
plt.figure(figsize=(10, 6))
sns.histplot(ratings_per_user, bins=50, log_scale=True)
plt.title('Number of Ratings per User')
plt.xlabel('Number of Ratings')
plt.ylabel('Count')
plt.show()

# Examine the number of ratings per book with log scale on x-axis
ratings_per_book = merged_df.groupby('ISBN').size()
plt.figure(figsize=(10, 6))
sns.histplot(ratings_per_book, bins=50, log_scale=True)
plt.title('Number of Ratings per Book')
plt.xlabel('Number of Ratings')
plt.ylabel('Count')
plt.show()


# In[6]:



# Check lengths of the arrays
actual_ratings_len = len(merged_df['Book-Rating'])
collab_len = len(collab_predictions)
content_len = len(content_predictions)
meta_len = len(y_pred_meta)


# In[7]:



# Check lengths of the arrays
print(actual_ratings_len)
print(collab_len)
print(content_len)
print(meta_len)


# In[5]:




# Function to plot rating distribution with counts
def plot_rating_distribution2(y_true, y_pred, model_name, rmse):
    bins = np.arange(1, 12) - 0.5
    true_counts, _ = np.histogram(y_true, bins=bins)
    pred_counts, _ = np.histogram(y_pred, bins=bins)
    bin_centers = 0.5 * (bins[1:] + bins[:-1])

    plt.figure(figsize=(12, 6))
    bar_width = 0.4

    plt.bar(bin_centers - bar_width / 2, true_counts, width=bar_width, color='blue', label='True Ratings')
    plt.bar(bin_centers + bar_width / 2, pred_counts, width=bar_width, color='red', label='Predicted Ratings')

    for i in range(len(bin_centers)):
        plt.text(bin_centers[i] - bar_width / 2, true_counts[i] + 50, str(true_counts[i]), ha='center', color='blue')
        plt.text(bin_centers[i] + bar_width / 2, pred_counts[i] + 50, str(pred_counts[i]), ha='center', color='red')

    plt.xlabel('Rating')
    plt.ylabel('Count')
    plt.title(f'Distribution of True Ratings and Predicted Ratings ({model_name}) - RMSE: {rmse:.4f}')
    plt.xticks(np.arange(1, 11))
    plt.legend()
    plt.show()

# Assuming the actual RMSE values and predictions are available
plot_rating_distribution2(merged_df['Book-Rating'], collab_predictions, 'Collaborative Filtering', collab_rmse)
plot_rating_distribution2(merged_df['Book-Rating'], content_predictions, 'Content-Based Filtering', content_rmse)
plot_rating_distribution2(y_val, y_pred_meta, 'Meta-Level Model', meta_rmse)




# In[21]:


# Function for Collaborative Filtering Model
def plot_pred_vs_actual_collab(y_true, y_pred, model_name, rmse):
    plt.figure(figsize=(12, 6))
    sns.boxplot(x=y_true, y=y_pred)
    plt.title(f'Predicted vs Actual Ratings (Collaborative Filtering) - RMSE: {rmse:.4f}')
    plt.xlabel('Actual Ratings')
    plt.ylabel('Predicted Ratings')
    plt.show()

# Function for Content-Based Filtering Model
def plot_pred_vs_actual_content(y_true, y_pred, model_name, rmse):
    plt.figure(figsize=(12, 6))
    sns.boxplot(x=y_true, y=y_pred)
    plt.title(f'Predicted vs Actual Ratings (Content-Based Filtering) - RMSE: {rmse:.4f}')
    plt.xlabel('Actual Ratings')
    plt.ylabel('Predicted Ratings')
    plt.show()

y_pred_meta2 = y_pred_meta.flatten()  # Flatten the array to 1D    
    
# Function for Meta-Level Model
def plot_pred_vs_actual_meta(y_true, y_pred, model_name, rmse):
    plt.figure(figsize=(12, 6))
    sns.boxplot(x=y_true, y=y_pred)
    plt.title(f'Predicted vs Actual Ratings (Meta-Level Model) - RMSE: {rmse:.4f}')
    plt.xlabel('Actual Ratings')
    plt.ylabel('Predicted Ratings')
    plt.show()


# In[22]:



plot_pred_vs_actual_collab(merged_df['Book-Rating'], collab_predictions, 'Collaborative Filtering', collab_rmse)
plot_pred_vs_actual_content(merged_df['Book-Rating'], content_predictions, 'Content-Based Filtering', content_rmse)
plot_pred_vs_actual_meta(y_val, y_pred_meta2, 'Meta-Level Model', meta_rmse)


# In[ ]:




