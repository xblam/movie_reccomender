import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error
import time
import pickle

class NNRecommender:
    # get and prepate the data
    def __init__(self, file_path):
        self.file_path = file_path
        self.model = None

        # getting the data set up
        self.df = self.load_data(file_path)
        self.X_train, self.X_test, self.y_train, self.y_test = self.load_data(self.file_path)

        # params for our grid search
        self.param_grid = {
            'hidden_layer_sizes': [(128,64)],  # Different layer sizes
            'activation': ['relu'],  # Different activation functions
            'solver': ['adam'],  # Optimization algorithm
            'alpha': [0.0001, 0.001, 0.01],  # Regularization strength
            'learning_rate': ['adaptive'],  # Learning rate schedules
            'max_iter': [1000]  # Number of iterations
        }

        # temp_params
        # self.param_grid = {
        #     'hidden_layer_sizes': [(64, 32)],  # Simplified layer sizes
        #     'activation': ['relu'],  # Keeping only 'relu' activation
        #     'alpha': [0.0001],  # Fewer regularization strengths to test
        #     'max_iter': [1000]  # Fixed number of iterations
        # }
    
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

    def train_model(self):
        # make model
        mlp = MLPRegressor(random_state=42)
        
        # gridcv search
        grid_search = GridSearchCV(mlp, self.param_grid, cv=5, scoring='neg_mean_absolute_error', verbose=2)
        grid_search.fit(self.X_train, self.y_train)

        # Print the best parameters and the best MAE
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best negative mean absolute error: {grid_search.best_score_}")

        # Set the best model
        self.model = grid_search.best_estimator_

        # Evaluate the model on the test set
        y_pred = self.model.predict(self.X_test)
        mae = mean_absolute_error(self.y_test, y_pred)
        print(f"MAE on test set: {mae}")

        with open('best_n_model.pkl', 'wb') as f:
            pickle.dump(self.model, f)


    def predict_ratings(self, input_file):
        """Predict ratings for the user-item pairs in the provided file"""
        df = pd.read_csv(input_file)
        
        # Extract user-item pairs from the input file
        X_input = df[['user_id', 'item_id']].values
        
        # Predict the ratings
        predictions = self.model.predict(X_input)
        
        # Return predictions as a 1D numpy array
        return predictions

if __name__ == "__main__":
    start_time = time.time()
    
    # Initialize the recommender with the dataset
    file_path = 'data_movie_lens_100k/ratings_all_development_set.csv'
    recommender = NNRecommender(file_path)
    recommender.load_data(file_path)
    
    # Train the model
    recommender.train_model()
    
    # Predict ratings on new data (e.g., masked ratings or leaderboard data)
    predictions = recommender.predict_ratings('data_movie_lens_100k/ratings_masked_leaderboard_set.csv')
    
    # Save predictions to a text file
    np.savetxt('predicted_ratings_leaderboard.txt', predictions, fmt='%.6f')
    
    # Print time taken
    end_time = time.time()
    total_time = end_time - start_time
    print(f"Total time taken: {total_time:.2f} seconds")