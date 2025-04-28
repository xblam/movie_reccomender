import pandas as pd
import numpy as np
from surprise import Dataset, Reader, SVD, accuracy
from surprise.model_selection import train_test_split, GridSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pickle
import time

class HybridModel:
    def __init__(self, file_path):
        self.file_path = file_path
        self.svd_model = None
        self.nn_model = None
        self.df = self.load_data(file_path)
        self.X_train, self.X_test, self.y_train, self.y_test = self.preprocess_data()

        # GridSearchCV for SVD parameters
        self.param_grid = {
            'n_factors': [10, 50, 100],  # Number of latent factors
            'n_epochs': [5, 10, 20],     # Number of epochs
            'lr_all': [0.001, 0.01],     # Learning rate
            'reg_all': [0.02, 0.1, 0.2]  # Regularization parameter
        }

    def load_data(self, file_path):
        # open up the file and extract values
        self.df = pd.read_csv(file_path)
        print(f"Loaded data with {self.df.shape[0]} entries.")
        print(self.df.head())

        # Extract user-item pairs and ratings
        X = self.df[['user_id', 'item_id']].values
        print(X.shape)
        y = self.df['rating'].values
        print(y.shape)
        
        # Split into training and testing data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        return X_train, X_test, y_train, y_test

    def load_data(self, file_path):
        """Load the dataset from the given file path"""
        df = pd.read_csv(file_path)
        reader = Reader(rating_scale=(1, 5))
        self.data = Dataset.load_from_df(df[['user_id', 'item_id', 'rating']], reader)

        """Preprocess data to extract user and item latent factors"""
        # Split data into train and test sets using Surprise's train_test_split
        trainset, testset = train_test_split(self.data, test_size=0.2)
        
        # Train SVD model (Matrix Factorization)
        svd = SVD()
        grid_search = GridSearchCV(svd, self.param_grid, measures=['rmse', 'mae'], cv=5)
        grid_search.fit(self.data)

        # Print best parameters from GridSearchCV for SVD
        print(f"Best parameters for SVD: {grid_search.best_params_}")
        best_svd_model = grid_search.best_estimator['rmse']

        # Extract user and item latent factors
        user_latent_factors = np.array([best_svd_model.pu[trainset.to_inner_uid(uid)] for uid in range(trainset.n_users)])
        item_latent_factors = np.array([best_svd_model.qi[trainset.to_inner_iid(iid)] for iid in range(trainset.n_items)])

        # Use the latent factors as features for the neural network
        X_train = user_latent_factors
        X_test = item_latent_factors
        y_train = np.array([rating for (_, _, rating) in trainset.all_ratings()])
        y_test = np.array([rating for (_, _, rating) in testset])

        return X_train, X_test, y_train, y_test

    def train_nn(self):
        """Train a Neural Network model (MLPRegressor) on the latent factors"""
        self.nn_model = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=1000, random_state=42)
        self.nn_model.fit(self.X_train, self.y_train)

    def evaluate(self):
        """Evaluate the hybrid model on the test set"""
        # Make predictions using the neural network model
        nn_predictions = self.nn_model.predict(self.X_test)

        # Calculate MAE and RMSE
        mae = mean_absolute_error(self.y_test, nn_predictions)
        rmse = np.sqrt(mean_squared_error(self.y_test, nn_predictions))

        print(f"MAE on test set: {mae}")
        print(f"RMSE on test set: {rmse}")

    def save_model(self):
        """Save the trained models"""
        with open('hybrid_svd_model.pkl', 'wb') as f:
            pickle.dump(self.svd_model, f)
        
        with open('hybrid_nn_model.pkl', 'wb') as f:
            pickle.dump(self.nn_model, f)

    def load_model(self):
        """Load the trained models"""
        with open('hybrid_svd_model.pkl', 'rb') as f:
            self.svd_model = pickle.load(f)
        
        with open('hybrid_nn_model.pkl', 'rb') as f:
            self.nn_model = pickle.load(f)

    def predict(self, user_id, item_id):
        """Predict the rating for a given user-item pair using the hybrid model"""
        # Get the latent factors for user and item
        user_latent = self.svd_model.pu[self.svd_model.trainset.to_inner_uid(user_id)]
        item_latent = self.svd_model.qi[self.svd_model.trainset.to_inner_iid(item_id)]

        # Use neural network to predict the rating
        prediction = self.nn_model.predict([np.concatenate((user_latent, item_latent))])

        return prediction[0]

if __name__ == "__main__":
    start_time = time.time()

    # Initialize the hybrid model
    hybrid_model = HybridModel('data_movie_lens_100k/ratings_all_development_set.csv')

    # Train the SVD (Matrix Factorization) model
    hybrid_model.train_nn()

    # Evaluate the performance of the model on the test set
    hybrid_model.evaluate()

    # Save the trained models
    hybrid_model.save_model()

    # Predict a rating for a given user-item pair
    user_id = 1  # Example user_id
    item_id = 10  # Example item_id
    prediction = hybrid_model.predict(user_id, item_id)
    print(f"Predicted rating for user {user_id} and item {item_id}: {prediction}")

    end_time = time.time()
    total_time = end_time - start_time
    print(f"Total time taken: {total_time:.2f} seconds")
