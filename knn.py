import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from train_valid_test_loader import load_train_valid_test_datasets

# Function to compute item-item similarity matrix
def compute_item_similarity(ratings_matrix):
    """
    Compute the item-item similarity matrix using cosine similarity
    Args:
        ratings_matrix (numpy array): User-item matrix (n_users x n_items)
    
    Returns:
        item_similarity_matrix (numpy array): Item-item similarity matrix (n_items x n_items)
    """
    # Normalize the ratings matrix by subtracting the mean rating per item
    ratings_matrix_normalized = ratings_matrix - np.mean(ratings_matrix, axis=0)
    
    # Compute the cosine similarity between items
    item_similarity_matrix = cosine_similarity(ratings_matrix_normalized.T)  # Transpose to calculate similarity between items
    return item_similarity_matrix

# Function to predict ratings using item-item similarity
def predict_ratings_knn(user_id, item_id, ratings_matrix, item_similarity_matrix, k=5):
    """
    Predict the rating for a specific user-item pair using KNN.
    Args:
        user_id (int): The user ID
        item_id (int): The item (movie) ID
        ratings_matrix (numpy array): The user-item rating matrix
        item_similarity_matrix (numpy array): The item-item similarity matrix
        k (int): The number of nearest neighbors to consider
    
    Returns:
        predicted_rating (float): The predicted rating for the user-item pair
    """
    # Get the ratings for the item from all users
    item_ratings = ratings_matrix[:, item_id]
    
    # Find the indices of the k most similar items
    similar_items = np.argsort(item_similarity_matrix[item_id])[-k:][::-1]
    
    # Get the ratings for the k most similar items
    similar_ratings = ratings_matrix[:, similar_items]
    
    # Calculate the weighted average of the ratings from the most similar items
    # Use the item similarities as weights
    similarities = item_similarity_matrix[item_id, similar_items]
    weighted_ratings = np.dot(similar_ratings.T, similarities)
    
    # Normalize the predicted rating based on the similarity scores
    predicted_rating = weighted_ratings / np.sum(similarities)
    
    return predicted_rating

# Function to train the KNN model
def train_knn_model(ratings_matrix, k=5):
    """
    Train the KNN model by computing the item-item similarity matrix.
    Args:
        ratings_matrix (numpy array): User-item rating matrix (n_users x n_items)
        k (int): The number of nearest neighbors to consider
    
    Returns:
        item_similarity_matrix (numpy array): The item-item similarity matrix (n_items x n_items)
    """
    # Compute the item-item similarity matrix
    item_similarity_matrix = compute_item_similarity(ratings_matrix)
    
    return item_similarity_matrix

# Example usage

# Load the dataset
train_tuple, valid_tuple, test_tuple, n_users, n_items = load_train_valid_test_datasets()

# Convert to numpy arrays (assuming the dataset is in the form (user_id, item_id, rating))
ratings_matrix = np.zeros((n_users, n_items))

# Fill the ratings_matrix with the provided ratings data
for user, item, rating in zip(train_tuple[0], train_tuple[1], train_tuple[2]):
    ratings_matrix[user, item] = rating

# Train the KNN model
k = 5  # Set the number of nearest neighbors
item_similarity_matrix = train_knn_model(ratings_matrix, k)

# Make predictions for a specific user-item pair
user_id = 10  # Example user
item_id = 20  # Example item (movie)

predicted_rating = predict_ratings_knn(user_id, item_id, ratings_matrix, item_similarity_matrix, k)
print(f"Predicted rating for user {user_id} and item {item_id}: {predicted_rating}")