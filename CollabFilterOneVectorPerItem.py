'''
CollabFilterOneVectorPerItem.py

Defines class: `CollabFilterOneVectorPerItem`

Scroll down to __main__ to see a usage example.
'''

# Make sure you use the autograd version of numpy (which we named 'ag_np')
# to do all the loss calculations, since automatic gradients are needed
import autograd.numpy as ag_np

# Use helper packages
from AbstractBaseCollabFilterSGD import AbstractBaseCollabFilterSGD
from train_valid_test_loader import load_train_valid_test_datasets

# Some packages you might need (uncomment as necessary)
import pandas as pd
import matplotlib.pyplot as plt

# No other imports specific to ML (e.g. scikit) needed!

class CollabFilterOneVectorPerItem(AbstractBaseCollabFilterSGD):
    ''' One-vector-per-user, one-vector-per-item recommendation model.

    Assumes each user, each item has learned vector of size `n_factors`.

    Attributes required in param_dict
    ---------------------------------
    mu : 1D array of size (1,)
    b_per_user : 1D array, size n_users
    c_per_item : 1D array, size n_items
    U : 2D array, size n_users x n_factors
    V : 2D array, size n_items x n_factors

    Notes
    -----
    Inherits *__init__** constructor from AbstractBaseCollabFilterSGD.
    Inherits *fit* method from AbstractBaseCollabFilterSGD.
    '''

    # takes in number of users, number of movies, 
    def init_parameter_dict(self, n_users, n_items, train_tuple):
        ''' Initialize parameter dictionary attribute for this instance.

        Post Condition
        --------------
        Updates the following attributes of this instance:
        * param_dict : dict
            Keys are string names of parameters
            Values are *numpy arrays* of parameter values
        '''
        random_state = self.random_state # inherited RandomState object

        # TODO fix the lines below to have right dimensionality & values
        # TIP: use self.n_factors to access number of hidden dimensions
        self.param_dict = dict(
            # global average rating
            mu=ag_np.ones(1),
            # lets us know how much the user overrates/underrates the movies
            b_per_user=ag_np.ones(n_users),
            # how much more or less do people like this movie
            c_per_item=ag_np.ones(n_items), 

            # vector representing user preferences
            U=0.001 * random_state.randn(n_users, self.n_factors),
            # vector representing movie traits, projected to the same traits
            V=0.001 * random_state.randn(n_items, self.n_factors), 
            )


    def predict(self, user_id_N, item_id_N,
                mu=None, b_per_user=None, c_per_item=None, U=None, V=None):
        ''' Predict ratings at specific user_id, item_id pairs

        Args
        ----
        user_id_N : 1D array, size n_examples
            Specific user_id values to use to make predictions
        item_id_N : 1D array, size n_examples
            Specific item_id values to use to make predictions
            Each entry is paired with the corresponding entry of user_id_N

        Returns
        -------
        yhat_N : 1D array, size n_examples
            Scalar predicted ratings, one per provided example.
            Entry n is for the n-th pair of user_id, item_id values provided.
        '''
        # Get user and movie vectosr, both will be (N, n_factors)
        user_vectors = U[user_id_N]
        item_vectors = V[item_id_N]

        # do dot product between user and item vectors
        interaction_scores = ag_np.sum(user_vectors * item_vectors, axis=1)  # shape (N,)

        # Get user and item biases
        user_biases = b_per_user[user_id_N]  # shape (N,)
        item_biases = c_per_item[item_id_N]  # shape (N,)

        # Combine everything to get the predicted ratings
        yhat_N = mu[0] + user_biases + item_biases + interaction_scores

        return yhat_N  # shape (N,)
    
    # data type is just the observed datasets
    def calc_loss_wrt_parameter_dict(self, param_dict, data_tuple):
        ''' Compute loss at given parameters

        Args
        ----
        param_dict : dict
            Keys are string names of parameters
            Values are *numpy arrays* of parameter values

        Returns
        -------
        loss : float scalar
        '''
        # Unpack true ratings
        y_N = data_tuple[2]

        # Predict ratings using current parameters
        yhat_N = self.predict(data_tuple[0], data_tuple[1], **param_dict)

        # Compute squared error loss
        squared_errors = ag_np.square(y_N - yhat_N)
        mse_loss = 0.5 * ag_np.mean(squared_errors)

        # Add l2 regularization on our terms
        U = param_dict['U']
        V = param_dict['V']
        l2_penalty = 0.5 * self.alpha * (ag_np.sum(U ** 2) + ag_np.sum(V ** 2))

        # Total loss = MSE + regularization
        loss_total = mse_loss + l2_penalty
        return loss_total   


if __name__ == '__main__':

    # Load the dataset
    train_tuple, valid_tuple, test_tuple, n_users, n_items = \
        load_train_valid_test_datasets()
    # Create the model and initialize its parameters


    # RUN FOR THE 3(i)
    # K_values = [2, 10, 50]
    K_values = [2]
    alpha_values = [0.0]

    results = []

    results = []

    for K in K_values:
        for alpha in alpha_values:
            print("=" * 50)
            print(f"Training model with K={K}, alpha={alpha}\n")
            
            model = CollabFilterOneVectorPerItem(
                n_epochs=10,
                batch_size=1000,  # DO NOT CHANGE
                step_size=0.1,
                n_factors=K,
                alpha=alpha
            )
            model.init_parameter_dict(n_users, n_items, train_tuple)
            model.fit(train_tuple, valid_tuple)

            val_metrics = model.evaluate_perf_metrics(*valid_tuple)
            test_metrics = model.evaluate_perf_metrics(*test_tuple)

            val_rmse = val_metrics["rmse"]
            val_mae = val_metrics["mae"]
            test_rmse = test_metrics["rmse"]
            test_mae = test_metrics["mae"]

            results.append({
                "K": K,
                "alpha": alpha,
                "val_rmse": val_rmse,
                "val_mae": val_mae,
                "test_rmse": test_rmse,
                "test_mae": test_mae,
                "trace_epoch": model.trace_epoch,
                "trace_rmse_train": model.trace_rmse_train,
                "trace_rmse_valid": model.trace_rmse_valid
            })
    for res in results:
        print(res["trace_epoch"])
    
    print("\n=== Final Scores for All Models ===\n")

    for res in results:
        print(f"K = {res['K']}, alpha = {res['alpha']} - "
            f"Validation RMSE: {res['val_rmse']}, MAE: {res['val_mae']} - "
            f"Test RMSE: {res['test_rmse']}, MAE: {res['test_mae']}")

    # get values for plotting
    trace_epoch_vals = [res["trace_epoch"] for res in results]
    trace_rmse_train_vals = [res["trace_rmse_train"] for res in results]
    trace_rmse_valid_vals = [res["trace_rmse_valid"] for res in results]
    print(trace_epoch_vals)
    print(trace_epoch_vals)

    plt.figure(figsize=(10, 6))
    plt.title("Validation RMSE vs Epoch")
    plt.xlabel("Number of Latent Factors (K)")
    plt.ylabel("Validation RMSE")
    plt.plot(trace_epoch_vals, trace_rmse_train_vals, marker='o', label='Train MAE')
    plt.plot(trace_epoch_vals, trace_rmse_valid_vals, label='Validation RMSE', marker='s')
    plt.legend()
    plt.show()
  #________________________________________________________________________________
    # to have right scale as the dataset (right num users and items)
    # model = CollabFilterOneVectorPerItem(
    #     n_epochs=10, batch_size=10000, step_size=0.1,
    #     n_factors=2, alpha=0.0)
    # model.init_parameter_dict(n_users, n_items, train_tuple)

    # # Fit the model with SGD
    # model.fit(train_tuple, valid_tuple)
