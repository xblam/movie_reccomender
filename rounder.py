import numpy as np

def round_and_save(input_file, output_file):
    # Load the data from the input file
    predictions = np.loadtxt(input_file)
    
    # Round each number to the nearest integer
    predictions_rounded = np.round(predictions).astype(int)
    
    # Clip the values to be within the range 1-5
    predictions_rounded = np.clip(predictions_rounded, 1, 5)
    
    # Save the rounded predictions to a new text file
    np.savetxt(output_file, predictions_rounded, fmt='%d')

# Example usage:
input_file = 'predicted_ratings_leaderboard.txt'  # Replace with your input file path
output_file = 'predicted_ratings_leaderboard.txt'  # Replace with your output file path
round_and_save(input_file, output_file)

print("Rounded predictions saved to:", output_file)