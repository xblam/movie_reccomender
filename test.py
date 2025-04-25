import numpy as np
# Number of predicted ratings
num_ratings = 10000

# Generate random predicted ratings between 1 and 5 (inclusive)
predicted_ratings = np.random.randint(1, 6, size=num_ratings)

# Filepath to save the ratings
file_path = "predicted_ratings_leaderboard.txt"

# Save the ratings to a text file, one rating per line
np.savetxt(file_path, predicted_ratings, fmt='%d')

print(f"File saved at: {file_path}")