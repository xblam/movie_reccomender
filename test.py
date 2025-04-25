import autograd.numpy as ag_np

# Random example with 3 users and 3 items, 2 latent factors each
user_vectors = ag_np.array([
    [1.0, 2.0],    # user 0
    [0.5, -1.0],   # user 1
    [-2.0, 0.0]    # user 2
])

item_vectors = ag_np.array([
    [3.0, 1.0],    # item 0
    [-1.0, 2.0],   # item 1
    [0.0, -1.0]    # item 2
])

# Compute interaction scores (dot products)
interaction_scores = ag_np.sum(user_vectors * item_vectors, axis=1)

print("User vectors:\n", user_vectors)
print("Item vectors:\n", item_vectors)
print("Dot products (interaction_scores):", interaction_scores)