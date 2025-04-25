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
        # self.param_dict = dict(
        #     mu = ag_np.ones(1) * (ag_np.mean(train_tuple[2])),
        #     # makes more sense for the bias to be set to something close to 0
        #     b_per_user = 0.001 * random_state.randn(n_users),  # Random biases for each user
        #     c_per_item = 0.001 * random_state.randn(n_items),  # Random biases for each item
        #     U=0.001 * random_state.randn(n_users, self.n_factors), # FIX dimensionality
        #     V=0.001 * random_state.randn(n_items, self.n_factors), # FIX dimensionality
        # )
        self.param_dict = dict(
            mu = ag_np.ones(1) * (ag_np.mean(train_tuple[2])),
            # makes more sense for the bias to be set to something close to 0
            b_per_user = ag_np.zeros(n_users),  # Random biases for each user
            c_per_item = ag_np.zeros(n_items),  # Random biases for each item
            U=0.001 * random_state.randn(n_users, self.n_factors), # FIX dimensionality
            V=0.001 * random_state.randn(n_items, self.n_factors), # FIX dimensionality
        )
        print(self.param_dict['mu'])
        print(self.param_dict['U'].shape)
        print(self.param_dict['V'].shape)


    def predict(self, user_id_N, item_id_N, mu=None, b_per_user=None, c_per_item=None, U=None, V=None):
        ''' Predict ratings at specific user_id, item_id pairs '''
        user_id_N = ag_np.array(user_id_N)
        item_id_N = ag_np.array(item_id_N)

        # Use vectorized computation for interaction scores
        user_vectors = U[user_id_N]  # Shape: (N, n_factors)
        item_vectors = V[item_id_N]  # Shape: (N, n_factors)

        # Interaction score as a dot product between user and item vectors
        interaction_scores = ag_np.sum(user_vectors * item_vectors, axis=1)  # Shape: (N,)

        # Vectorized bias computation
        user_biases = b_per_user[user_id_N]  # Shape: (N,)
        item_biases = c_per_item[item_id_N]  # Shape: (N,)

        # Final predicted ratings: μ + b_user + c_item + interaction
        yhat_N = mu + user_biases + item_biases + interaction_scores
        return yhat_N


    def calc_loss_wrt_parameter_dict(self, param_dict, data_tuple):
        y_N = data_tuple[2]  # True ratings (numpy array)

        # Get the predicted ratings using the current model parameters
        yhat_N = self.predict(data_tuple[0], data_tuple[1], **param_dict)

        # Compute squared error loss (SUM of squared errors)
        squared_errors = ag_np.square(y_N - yhat_N)  # Compute element-wise squared errors
        loss_total = ag_np.sum(squared_errors)  # Use SUM instead of MEAN

        # Regularization: L2 penalty for U and V
        U = param_dict['U']
        V = param_dict['V']

        # Compute L2 regularization penalty
        l2_penalty = 0.5 * self.alpha * (ag_np.sum(U ** 2) + ag_np.sum(V ** 2))

        # Total loss = SUM of squared errors + regularization
        loss_total += l2_penalty

        return loss_total
 



if __name__ == '__main__':

    # Load the dataset
    train_tuple, valid_tuple, test_tuple, n_users, n_items = \
        load_train_valid_test_datasets()

    K_values = [10] # turn this into a loop later
    alpha = 0.0
    for K in K_values:
        # Create an instance of the model
        model = CollabFilterOneVectorPerItem(
            n_epochs=100,               # Number of epochs for training
            batch_size=1000,           # Fixed batch size of 1000
            step_size=0.5,             # Learning rate (you can adjust this if needed)
            n_factors=K,               # Set the number of latent factors to K
            alpha=alpha                 # Set regularization strength to 0
        )


        model.init_parameter_dict(n_users, n_items, train_tuple)

        # Fit the model with SGD
        model.fit(train_tuple, valid_tuple)

        plt.plot(model.trace_epoch, model.trace_rmse_train, '-')
        plt.plot(model.trace_epoch, model.trace_rmse_valid, '-')
        plt.title(f'RMSE vs Epoch for K={K}')
        plt.xlabel('Epoch')
        plt.ylabel('RMSE')
        plt.legend(['Train RMSE', 'Validation RMSE'])
        plt.show()

