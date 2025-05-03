import pandas as pd
# Load the dataset
file_path = 'movie_dataset.csv'
movie_dataset = pd.read_csv(file_path)

# Keep only the required columns
columns_to_keep = ['index', 'genres', 'original_title', 'original_language']
columns_to_keep.extend(['director', 'cast', 'release_date', 'runtime', 'vote_count', 'revenue', 'overview'])
filtered_dataset = movie_dataset[columns_to_keep]

# Save the filtered dataset back to a CSV file
filtered_file_path = 'filtered_movie_dataset.csv'
filtered_dataset.to_csv(filtered_file_path, index=False)

# now cleaning the filtered dataset
def clean_data(df):
    # Remove duplicates
    df = df.drop_duplicates()
    
    # Remove rows with missing values
    df = df.dropna()
    
    # Reset index
    df = df.reset_index(drop=True)
    
    return df

# Clean the filtered dataset
cleaned_dataset = clean_data(filtered_dataset)
