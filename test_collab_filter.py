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
            mu = ag_np.ones(1) * (ag_np.mean(train_tuple[2])),
            # makes more sense for the bias to be set to something close to 0
            b_per_user = 0.001 * random_state.randn(n_users),  # Random biases for each user
            c_per_item = 0.001 * random_state.randn(n_items),  # Random biases for each item
            U=0.001 * random_state.randn(n_users, self.n_factors), # FIX dimensionality
            V=0.001 * random_state.randn(n_items, self.n_factors), # FIX dimensionality
        )
        print(self.param_dict['mu'])

    def predict(self, user_id_N, item_id_N,
                mu=None, b_per_user=None, c_per_item=None, U=None, V=None):
        
        ''' Predict ratings at specific user_id, item_id pairs '''
        user_id_N = ag_np.array(user_id_N)
        item_id_N = ag_np.array(item_id_N)

        user_vectors = U[user_id_N]  # Shape: (N, n_factors)
        item_vectors = V[item_id_N]  # Shape: (N, n_factors)

        interaction_scores = ag_np.sum(user_vectors * item_vectors, axis=1)  # Shape: (N,)

        user_biases = b_per_user[user_id_N]  # Shape: (N,)
        item_biases = c_per_item[item_id_N]  # Shape: (N,)

        yhat_N = mu + user_biases + item_biases + interaction_scores
        return yhat_N



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
        # TODO compute loss
        # TIP: use self.alpha to access regularization strength
            # Unpack the true ratings from data_tuple
        y_N = data_tuple[2]  # True ratings (numpy array)

        # Get the predicted ratings using the current model parameters
        yhat_N = self.predict(data_tuple[0], data_tuple[1], **param_dict)

        # Compute squared error loss (MSE)
        squared_errors = ag_np.square(y_N - yhat_N)  # Compute element-wise squared errors
        mse_loss = ag_np.mean(squared_errors)  # Mean squared error

        # Regularization: L2 penalty for U and V
        U = param_dict['U']
        V = param_dict['V']

        # Compute L2 regularization penalty
        l2_penalty = 0.5 * self.alpha * (ag_np.sum(U ** 2) + ag_np.sum(V ** 2))

        # Total loss = MSE + regularization
        loss_total = mse_loss + l2_penalty

        return loss_total



if __name__ == '__main__':
    # Load the dataset
    train_tuple, valid_tuple, test_tuple, n_users, n_items = \
        load_train_valid_test_datasets()

    # Set the value of K (number of latent factors)
    K_values = [2, 10, 50]  # You can set this to any value, for now just one at a time
    alpha = 0.1  # Regularization strength (set as needed)

    results = {}  # Store results for all K values, as a dictionary

    # Loop over each value of K (optional for later testing multiple K)
    for K in K_values:
        print(f"\nTraining model with K = {K}, alpha = {alpha}")

        # Create the model with the specified parameters
        model = CollabFilterOneVectorPerItem(
            n_epochs=100,               # Number of epochs for training
            batch_size=1000,           # Fixed batch size of 1000
            step_size=0.1,             # Learning rate
            n_factors=50,               # Set the number of latent factors to K
            alpha=alpha                # Set regularization strength
        )

        # Initialize model parameters
        model.init_parameter_dict(n_users, n_items, train_tuple)

        # Fit the model with SGD
        model.fit(train_tuple, valid_tuple)

        # Initialize an empty list for this K value in results dictionary
        if K not in results:
            results[K] = []  # Create a list for each K if it doesn't exist

        # Store the results (trace data for RMSE)
        results[K].append({
            "K": K,
            "trace_epoch": model.trace_epoch,
            "trace_rmse_train": model.trace_rmse_train,
            "trace_rmse_valid": model.trace_rmse_valid,
        })

        # Optionally, evaluate and print final results (validation/test performance)
        val_metrics = model.evaluate_perf_metrics(*valid_tuple)
        test_metrics = model.evaluate_perf_metrics(*test_tuple)

        print(f"Validation RMSE: {val_metrics['rmse']}, Validation MAE: {val_metrics['mae']}")
        print(f"Test RMSE: {test_metrics['rmse']}, Test MAE: {test_metrics['mae']}")

    # After training, generate the RMSE vs. epoch plot for a single value of K
    K = 50  # Choose the value of K you want to plot

    # Extract the trace data for the current value of K
    trace_rmse_train = [res["trace_rmse_train"] for res in results[K]][0]
    trace_rmse_valid = [res["trace_rmse_valid"] for res in results[K]][0]
    trace_epoch = [res["trace_epoch"] for res in results[K]][0]

    # Plot the RMSE for train and validation
    plt.figure(figsize=(8, 6))
    plt.plot(trace_epoch, trace_rmse_train, label="Train RMSE", color="blue")
    plt.plot(trace_epoch, trace_rmse_valid, label="Validation RMSE", color="orange")

    # Set titles, labels, and legends
    plt.title(f'RMSE vs Epoch for K={K}')
    plt.xlabel('Epoch')
    plt.ylabel('RMSE')
    plt.legend()
    plt.grid(True)

    # Show the plot
    plt.tight_layout()
    plt.show()