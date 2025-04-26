
import pandas as pd
from surprise import Dataset, Reader, SVD, accuracy
from surprise.model_selection import train_test_split, GridSearchCV
import numpy as np
import pickle
import time

class SVDModel:
    
    def __init__(self, file_path):
        self.file_path = file_path
        self.model = None
        self.data = self.load_data(file_path)
        self.param_grid = {
            'n_factors': [2, 10, 50],  # Number of latent factors
            'n_epochs': [10, 100],    # Number of epochs
            'lr_all': [0.001, 0.01, 0.1],  # Learning rate
            'reg_all': [0.02, 0.1, 0.2],     # Regularization parameter
            # 'n_factors': [2],  # Number of latent factors
            # 'n_epochs': [10],    # Number of epochs
            # 'lr_all': [0.01],  # Learning rate
            # 'reg_all': [0.02],     # Regularization parameter
        }
    
    def load_data(self, file_path):
        """Load the dataset from the given file path"""
        df = pd.read_csv(file_path)
        reader = Reader(rating_scale=(1, 5))
        data = Dataset.load_from_df(df[['user_id', 'item_id', 'rating']], reader)
        
        # Build the full trainset to access the number of ratings
        trainset = data.build_full_trainset()

        # Print the size of the dataset (number of ratings)
        print(f"Number of ratings: {trainset.n_ratings}")
        print(f"Number of users: {trainset.n_users}")
        print(f"Number of items: {trainset.n_items}")
        
        return data


    def train_model(self):
        """Train the model using GridSearchCV and save the best model."""
        # Use GridSearchCV to perform hyperparameter tuning
        grid_search = GridSearchCV(SVD, self.param_grid, measures=['rmse', 'mae'], cv=5)

        # Perform the grid search with 5-fold cross-validation
        grid_search.fit(self.data)

        # Print the best parameters and the corresponding RMSE and MAE
        print(f"Best parameters: {grid_search.best_params}")
        print(f"Best RMSE: {grid_search.best_score['rmse']}")
        print(f"Best MAE: {grid_search.best_score['mae']}")

        # Retrieve the best model
        self.model = grid_search.best_estimator['mae']

        # Make predictions with the best model
        trainset = self.data.build_full_trainset()
        self.model.fit(trainset)

        # Save the best model using pickle
        with open('best_svd_model.pkl', 'wb') as f:
            pickle.dump(self.model, f)

    def predict_ratings(self, input_file):
        """Predict ratings for the user-item pairs in the provided file.
           Returns predictions as a 1D array of floats with shape (10000,)."""
        df = pd.read_csv(input_file)
        
        # We don't have ratings in the input, just user_id, item_id
        user_ids = df['user_id'].values
        item_ids = df['item_id'].values

        # Predict the ratings
        predictions = []
        for user_id, item_id in zip(user_ids, item_ids):
            pred = self.model.predict(user_id, item_id)
            predictions.append(pred.est)  # Append predicted rating

        # Convert predictions to a 1D numpy array of floats
        predicted_ratings = np.array(predictions, dtype=float)

        return predicted_ratings



def load_trained_model():
    """Load the trained model from the pickle file."""
    with open('best_svd_model.pkl', 'rb') as f:
        model = pickle.load(f)
    return model

if __name__ == "__main__":
    # Time the training and prediction process
    start_time = time.time()

    # Initialize the SVD model class with the dataset path
    svd_model = SVDModel('data_movie_lens_100k/ratings_all_development_set.csv')

    # Train the model
    svd_model.train_model()
    svd_model.model = load_trained_model()

    # Make predictions on new data (e.g., masked ratings)
    predicted_ratings = svd_model.predict_ratings('data_movie_lens_100k/ratings_masked_leaderboard_set.csv')
    print(predicted_ratings.shape)
    np.savetxt('predicted_ratings_leaderboard.txt', predicted_ratings, fmt='%.6f')



    # Check the total time taken for the process
    end_time = time.time()
    total_time = end_time - start_time
    print(f"Total time taken: {total_time:.2f} seconds")