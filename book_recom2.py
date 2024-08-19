#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
from surprise import Dataset, Reader, SVD, accuracy
from surprise.model_selection import cross_validate, KFold as SurpriseKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold
from keras_tuner import RandomSearch
from annoy import AnnoyIndex
from tqdm import tqdm
import matplotlib.pyplot as plt

# Load merged data
merged_data_path = 'C:\\Users\\sethb\\Documents\\Capstonemerged_data.csv'
merged_df = pd.read_csv(merged_data_path)

# Data Preprocessing: Handle missing values and normalize
merged_df = merged_df.fillna('')
merged_df['Book-Rating'] = merged_df['Book-Rating'].astype(float)

# Split the data into training+validation and test sets
train_val_df, test_df = train_test_split(merged_df, test_size=0.2, random_state=42)

# Further split training+validation set into training and validation sets
cf_train_df, cf_val_df = train_test_split(train_val_df, test_size=0.2, random_state=42)

# Step 1: Build and Evaluate Collaborative Filtering Model with Cross-Validation
reader = Reader(rating_scale=(1, 10))
data = Dataset.load_from_df(cf_train_df[['User-ID', 'ISBN', 'Book-Rating']], reader)
collab_svd = SVD()

# Define evaluation metrics
def evaluate_model(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    return rmse, mae

# Cross-validate the collaborative filtering model
surprise_kf = SurpriseKFold(n_splits=5)
cv_results = cross_validate(collab_svd, data, measures=['RMSE', 'MAE'], cv=surprise_kf, verbose=True)
print(f'Collaborative Filtering Cross-Validation RMSE: {np.mean(cv_results["test_rmse"])}')
print(f'Collaborative Filtering Cross-Validation MAE: {np.mean(cv_results["test_mae"])}')

# Fit on full training data
trainset = data.build_full_trainset()
collab_svd.fit(trainset)

valset_cf = [(uid, iid, r) for (uid, iid, r) in cf_val_df[['User-ID', 'ISBN', 'Book-Rating']].values]
val_predictions_cf = collab_svd.test(valset_cf)
collab_val_rmse = accuracy.rmse(val_predictions_cf)
collab_val_mae = accuracy.mae(val_predictions_cf)
print(f'Collaborative Filtering Model Validation RMSE: {collab_val_rmse}')
print(f'Collaborative Filtering Model Validation MAE: {collab_val_mae}')

cf_val_preds = np.array([pred.est for pred in val_predictions_cf])

# Step 2: Build and Evaluate Content-Based Filtering Model with Cross-Validation
cf_train_df['Combined-Features'] = cf_train_df['Book-Title'] + ' ' + cf_train_df['Book-Author'] + ' ' + cf_train_df['Publisher']
cf_val_df['Combined-Features'] = cf_val_df['Book-Title'] + ' ' + cf_val_df['Book-Author'] + ' ' + cf_val_df['Publisher']
tfidf = TfidfVectorizer(stop_words='english')
svd = TruncatedSVD(n_components=50)

# Function to get content-based predictions
def get_content_predictions(df, svd_matrix, isbn_to_index):
    annoy_index = AnnoyIndex(50, 'angular')
    for i in range(len(svd_matrix)):
        annoy_index.add_item(i, svd_matrix[i])
    annoy_index.build(10)
    
    def predict(isbn):
        if isbn in isbn_to_index:
            neighbors = annoy_index.get_nns_by_vector(svd_matrix[isbn_to_index[isbn]], 10)
            ratings = [cf_train_df.iloc[neighbor]['Book-Rating'] for neighbor in neighbors]
            if len(ratings) > 0:
                return np.mean(ratings)
        return np.nan
    
    predictions = []
    for row in tqdm(df.itertuples(index=False), total=len(df)):
        isbn = row.ISBN
        prediction = predict(isbn)
        predictions.append(prediction)
    
    return np.array(predictions)

kf = KFold(n_splits=5)
rmse_scores = []
mae_scores = []

for train_index, val_index in kf.split(cf_train_df):
    train_fold, val_fold = cf_train_df.iloc[train_index], cf_train_df.iloc[val_index]
    
    # Combine features for train and validation folds
    train_fold['Combined-Features'] = train_fold['Book-Title'] + ' ' + train_fold['Book-Author'] + ' ' + train_fold['Publisher']
    val_fold['Combined-Features'] = val_fold['Book-Title'] + ' ' + val_fold['Book-Author'] + ' ' + val_fold['Publisher']
    
    # TF-IDF and SVD
    tfidf_matrix_train_fold = tfidf.fit_transform(train_fold['Combined-Features'])
    tfidf_matrix_val_fold = tfidf.transform(val_fold['Combined-Features'])
    svd_matrix_train_fold = svd.fit_transform(tfidf_matrix_train_fold)
    svd_matrix_val_fold = svd.transform(tfidf_matrix_val_fold)
    
    # Generate content-based predictions
    isbn_to_index_fold = {isbn: index for index, isbn in enumerate(train_fold['ISBN'])}
    mean_rating = train_fold['Book-Rating'].mean()
    cb_val_fold_preds = get_content_predictions(val_fold, svd_matrix_train_fold, isbn_to_index_fold)
    cb_val_fold_preds = np.where(np.isnan(cb_val_fold_preds), mean_rating, cb_val_fold_preds)
    
    # Evaluate
    rmse, mae = evaluate_model(val_fold['Book-Rating'].values, cb_val_fold_preds)
    rmse_scores.append(rmse)
    mae_scores.append(mae)

print(f'Content-Based Filtering Cross-Validation RMSE: {np.mean(rmse_scores)}')
print(f'Content-Based Filtering Cross-Validation MAE: {np.mean(mae_scores)}')

# Fit final content-based model on full training data
tfidf_matrix_train = tfidf.fit_transform(cf_train_df['Combined-Features'])
svd_matrix_train = svd.fit_transform(tfidf_matrix_train)
isbn_to_index = {isbn: index for index, isbn in enumerate(cf_train_df['ISBN'])}

# Generate content-based predictions for validation set
cb_val_preds = get_content_predictions(cf_val_df, svd_matrix_train, isbn_to_index)
mean_rating = cf_train_df['Book-Rating'].mean()
cb_val_preds = np.where(np.isnan(cb_val_preds), mean_rating, cb_val_preds)

# Ensure correct mapping for validation
print(f"First 10 actual ratings (validation): {cf_val_df['Book-Rating'].values[:10]}")
print(f"First 10 predicted ratings (validation): {cb_val_preds[:10]}")

# Generate content-based predictions for test set
cb_test_preds = get_content_predictions(test_df, svd_matrix_train, isbn_to_index)
cb_test_preds = np.where(np.isnan(cb_test_preds), mean_rating, cb_test_preds)

# Ensure correct mapping for test set
print(f"First 10 actual ratings (test): {test_df['Book-Rating'].values[:10]}")
print(f"First 10 predicted ratings (test): {cb_test_preds[:10]}")

# Standardize the features for meta model
scaler = StandardScaler()
X_meta = np.vstack((cf_val_preds, cb_val_preds)).T
X_meta = scaler.fit_transform(X_meta)
y_meta = cf_val_df['Book-Rating'].values

# Step 3: Build and Evaluate Meta-Model with Hyperparameter Tuning
def build_model(hp):
    model = Sequential()
    model.add(Dense(units=hp.Int('units1', min_value=32, max_value=512, step=32), activation='relu', input_shape=(X_meta.shape[1],)))
    model.add(Dropout(rate=hp.Float('dropout1', min_value=0.0, max_value=0.5, step=0.1)))
    model.add(Dense(units=hp.Int('units2', min_value=32, max_value=512, step=32), activation='relu'))
    model.add(Dropout(rate=hp.Float('dropout2', min_value=0.0, max_value=0.5, step=0.1)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

tuner = RandomSearch(
    build_model,
    objective='val_loss',
    max_trials=10,
    executions_per_trial=3,
    directory='my_dir',
    project_name='tuning'
)

tuner.search(X_meta, y_meta, epochs=10, validation_split=0.2)

best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
print(f"Best hyperparameters: {best_hps}")

meta_model = tuner.hypermodel.build(best_hps)
history = meta_model.fit(X_meta, y_meta, epochs=20, batch_size=32, validation_split=0.2)

# Evaluate on Test Set (ensure test set is unseen until now)
test_df['Combined-Features'] = test_df['Book-Title'] + ' ' + test_df['Book-Author'] + ' ' + test_df['Publisher']
tfidf_matrix_test = tfidf.transform(test_df['Combined-Features'])
svd_matrix_test = svd.transform(tfidf_matrix_test)

# Collaborative Filtering Predictions on Test Set
testset_cf = [(uid, iid, r) for (uid, iid, r) in test_df[['User-ID', 'ISBN', 'Book-Rating']].values]
test_predictions_cf = collab_svd.test(testset_cf)
cf_test_preds = np.array([pred.est for pred in test_predictions_cf])
collab_test_rmse, collab_test_mae = evaluate_model(test_df['Book-Rating'].values, cf_test_preds)
print(f'Collaborative Filtering Model Test RMSE: {collab_test_rmse}')
print(f'Collaborative Filtering Model Test MAE: {collab_test_mae}')

# Content-Based Filtering Predictions on Test Set
cb_test_preds = get_content_predictions(test_df, svd_matrix_train, isbn_to_index)
cb_test_preds = np.where(np.isnan(cb_test_preds), mean_rating, cb_test_preds)
cb_test_rmse, cb_test_mae = evaluate_model(test_df['Book-Rating'].values, cb_test_preds)
print(f'Content-Based Filtering Model Test RMSE: {cb_test_rmse}')
print(f'Content-Based Filtering Model Test MAE: {cb_test_mae}')

# Meta-Model Predictions on Test Set
X_test_meta = np.vstack((cf_test_preds, cb_test_preds)).T
X_test_meta = scaler.transform(X_test_meta)
y_test_meta = test_df['Book-Rating'].values

meta_test_predictions = meta_model.predict(X_test_meta)
meta_test_rmse, meta_test_mae = evaluate_model(y_test_meta, meta_test_predictions)
print(f'Meta-Model (Neural Network) Test RMSE: {meta_test_rmse}')
print(f'Meta-Model (Neural Network) Test MAE: {meta_test_mae}')

# Plot training history
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='validation')
plt.title('Meta-Model Training History')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Visualization of Model Performance
models = ['Collaborative Filtering', 'Content-Based Filtering', 'Meta-Model']
rmse_scores = [collab_test_rmse, cb_test_rmse, meta_test_rmse]
mae_scores = [collab_test_mae, cb_test_mae, meta_test_mae]

# RMSE Comparison
plt.figure(figsize=(10, 5))
plt.bar(models, rmse_scores, color=['blue', 'green', 'red'])
plt.title('Test RMSE Comparison')
plt.ylabel('RMSE')
plt.show()

# MAE Comparison
plt.figure(figsize=(10, 5))
plt.bar(models, mae_scores, color=['blue', 'green', 'red'])
plt.title('Test MAE Comparison')
plt.ylabel('MAE')
plt.show()

# Distribution of True Ratings and Predicted Ratings
def plot_distribution(y_true, y_pred, model_name, rmse):
    plt.figure(figsize=(12, 6))
    plt.hist(y_true, bins=np.arange(1, 12) - 0.5, alpha=0.5, label='True Ratings', color='blue', rwidth=0.8)
    plt.hist(y_pred, bins=np.arange(1, 12) - 0.5, alpha=0.5, label='Predicted Ratings', color='red', rwidth=0.8)
    plt.title(f'Distribution of True Ratings and Predicted Ratings ({model_name}) - RMSE: {rmse:.4f}')
    plt.xlabel('Rating')
    plt.ylabel('Count')
    plt.legend()
    plt.show()

# Box Plot of True Ratings vs. Predicted Ratings
def plot_box(y_true, y_pred, model_name, rmse):
    plt.figure(figsize=(12, 6))
    df = pd.DataFrame({'True Ratings': y_true, 'Predicted Ratings': y_pred})
    df.boxplot(by='True Ratings', column='Predicted Ratings', grid=False)
    plt.title(f'Predicted vs Actual Ratings ({model_name}) - RMSE: {rmse:.4f}')
    plt.suptitle('')
    plt.xlabel('Actual Ratings')
    plt.ylabel('Predicted Ratings')
    plt.show()

# Collaborative Filtering
plot_distribution(test_df['Book-Rating'].values, cf_test_preds, 'Collaborative Filtering', collab_test_rmse)
plot_box(test_df['Book-Rating'].values, cf_test_preds, 'Collaborative Filtering', collab_test_rmse)

# Content-Based Filtering
plot_distribution(test_df['Book-Rating'].values, cb_test_preds, 'Content-Based Filtering', cb_test_rmse)
plot_box(test_df['Book-Rating'].values, cb_test_preds, 'Content-Based Filtering', cb_test_rmse)

# Meta-Model
plot_distribution(test_df['Book-Rating'].values, meta_test_predictions.flatten(), 'Meta-Model', meta_test_rmse)
plot_box(test_df['Book-Rating'].values, meta_test_predictions.flatten(), 'Meta-Model', meta_test_rmse)


# In[5]:


def plot_scatter_with_line(y_true, y_pred, model_name, rmse):
    plt.figure(figsize=(8, 8))
    plt.scatter(y_true, y_pred, alpha=0.3, label='Predicted vs Actual')
    plt.plot([1, 10], [1, 10], 'r--', label='Perfect Agreement', linewidth=2)
    plt.xlabel('Actual Ratings')
    plt.ylabel('Predicted Ratings')
    plt.title(f'Scatter Plot of Actual vs Predicted Ratings ({model_name}) - RMSE: {rmse:.4f}')
    plt.legend()
    plt.grid(True)
    plt.show()

# Collaborative Filtering
plot_scatter_with_line(test_df['Book-Rating'].values, cf_test_preds, 'Collaborative Filtering', collab_test_rmse)

# Content-Based Filtering
plot_scatter_with_line(test_df['Book-Rating'].values, cb_test_preds, 'Content-Based Filtering', cb_test_rmse)

# Meta-Model
plot_scatter_with_line(test_df['Book-Rating'].values, meta_test_predictions.flatten(), 'Meta-Model', meta_test_rmse)


# In[6]:


def plot_error_distribution(y_true, y_pred, model_name):
    errors = y_pred - y_true
    plt.figure(figsize=(10, 6))
    plt.hist(errors, bins=30, alpha=0.7, color='blue', edgecolor='black')
    plt.title(f'Error Distribution Histogram ({model_name})')
    plt.xlabel('Prediction Error')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()

# Collaborative Filtering
plot_error_distribution(test_df['Book-Rating'].values, cf_test_preds, 'Collaborative Filtering')

# Content-Based Filtering
plot_error_distribution(test_df['Book-Rating'].values, cb_test_preds, 'Content-Based Filtering')

# Meta-Model
plot_error_distribution(test_df['Book-Rating'].values, meta_test_predictions.flatten(), 'Meta-Model')


# In[7]:


def plot_cumulative_error_distribution(y_true, y_pred, model_name):
    errors = np.abs(y_pred - y_true)
    sorted_errors = np.sort(errors)
    cumulative_errors = np.cumsum(sorted_errors)
    cumulative_errors = cumulative_errors / cumulative_errors[-1]
    
    plt.figure(figsize=(10, 6))
    plt.plot(sorted_errors, cumulative_errors, marker='.', linestyle='none')
    plt.title(f'Cumulative Error Distribution ({model_name})')
    plt.xlabel('Absolute Error')
    plt.ylabel('Cumulative Percentage')
    plt.grid(True)
    plt.show()

# Collaborative Filtering
plot_cumulative_error_distribution(test_df['Book-Rating'].values, cf_test_preds, 'Collaborative Filtering')

# Content-Based Filtering
plot_cumulative_error_distribution(test_df['Book-Rating'].values, cb_test_preds, 'Content-Based Filtering')

# Meta-Model
plot_cumulative_error_distribution(test_df['Book-Rating'].values, meta_test_predictions.flatten(), 'Meta-Model')


# In[8]:


def plot_bland_altman(y_true, y_pred, model_name):
    mean_ratings = np.mean([y_true, y_pred], axis=0)
    differences = y_pred - y_true
    mean_diff = np.mean(differences)
    std_diff = np.std(differences)
    
    plt.figure(figsize=(10, 6))
    plt.scatter(mean_ratings, differences, alpha=0.5)
    plt.axhline(mean_diff, color='red', linestyle='--', label=f'Mean Difference: {mean_diff:.2f}')
    plt.axhline(mean_diff + 1.96 * std_diff, color='gray', linestyle='--')
    plt.axhline(mean_diff - 1.96 * std_diff, color='gray', linestyle='--')
    plt.title(f'Bland-Altman Plot ({model_name})')
    plt.xlabel('Mean of Actual and Predicted Ratings')
    plt.ylabel('Difference between Predicted and Actual Ratings')
    plt.legend()
    plt.grid(True)
    plt.show()

# Collaborative Filtering
plot_bland_altman(test_df['Book-Rating'].values, cf_test_preds, 'Collaborative Filtering')

# Content-Based Filtering
plot_bland_altman(test_df['Book-Rating'].values, cb_test_preds, 'Content-Based Filtering')

# Meta-Model
plot_bland_altman(test_df['Book-Rating'].values, meta_test_predictions.flatten(), 'Meta-Model')


# In[10]:


def plot_heatmap(y_true, y_pred, model_name):
    heatmap, xedges, yedges = np.histogram2d(y_true, y_pred, bins=(np.arange(1, 12) - 0.5, np.arange(1, 12) - 0.5))
    plt.figure(figsize=(10, 8))
    plt.imshow(heatmap.T, origin='lower', cmap='hot', interpolation='nearest')
    plt.colorbar()
    plt.xlabel('Actual Ratings')
    plt.ylabel('Predicted Ratings')
    plt.title(f'Predicted vs Actual Rating Heatmap ({model_name})')
    plt.show()

# Collaborative Filtering
plot_heatmap(test_df['Book-Rating'].values, cf_test_preds, 'Collaborative Filtering')

# Content-Based Filtering
plot_heatmap(test_df['Book-Rating'].values, cb_test_preds, 'Content-Based Filtering')

# Meta-Model
plot_heatmap(test_df['Book-Rating'].values, meta_test_predictions.flatten(), 'Meta-Model')


# In[ ]:




