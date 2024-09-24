import os

from surprise import Dataset, Reader
from surprise import accuracy
import pandas as pd


def test_recommender(algo, data_path, output_path, index, n_recent_entries, heading_name):
    print(f"Running test with index {index} to produce {heading_name}")

    # Paths for base and test files
    train_file = data_path + f'/u{index}.base'
    test_file = data_path + f'/u{index}.test'

    # Load the train and test data using dataframes
    train_df = pd.read_csv(train_file, sep='\t', names=['userId', 'movieId', 'rating', 'timestamp'])
    test_df = pd.read_csv(test_file, sep='\t', names=['userId', 'movieId', 'rating', 'timestamp'])

    # Reduce the data to the n most recent if it is a positive number, otherwise use the whole dataset
    if n_recent_entries > 0:
        # Sort by userId and timestamp to ensure recent entries are at the end
        train_df.sort_values(by=['userId', 'timestamp'], inplace=True)

        # Group by userId and select the n most recent entries for each user
        train_df = train_df.groupby('userId').tail(n_recent_entries)

    # Define a Reader object with the appropriate rating scale
    reader = Reader(rating_scale=(1, 5))

    # Load the data into Surprise's Dataset object
    train_data = Dataset.load_from_df(train_df[['userId', 'movieId', 'rating']], reader)
    train_set = train_data.build_full_trainset()

    # Convert test data into Surprise's test_set format
    test_set = list(test_df[['userId', 'movieId', 'rating']].itertuples(index=False, name=None))

    print("Training...")
    # Train the algorithm on the training set
    algo.fit(train_set)

    # Make predictions on the test set
    print("Testing...")
    predictions = algo.test(test_set)

    # Calculate and print Root Mean Squared Error for the test set
    rmse = accuracy.rmse(predictions)
    print(f"RMSE: {rmse}")

    # Prepare a DataFrame for the predictions
    predictions_df = pd.DataFrame(predictions, columns=['user_id', 'movie_id', 'r_ui', heading_name, 'details'])

    # Remove unnecessary columns
    predictions_df = predictions_df.drop(columns=['r_ui', 'details'])

    # Load existing csv file
    existing_csv_file_path = output_path + f'/rating_test_df_u{index}.csv'

    # Check if the file exists
    if os.path.exists(existing_csv_file_path):
        # Load existing csv file
        output_df = pd.read_csv(existing_csv_file_path)

        # Remove previous test data if it exists
        if heading_name in output_df.columns:
            output_df = output_df.drop(columns=[heading_name])
    else:
        # Create a new DataFrame with the same columns as predictions_df
        output_df = test_df.rename(columns={
                        'userId': 'user_id',
                        'movieId': 'movie_id'
                    })
        output_df.insert(0, '', range(1, len(output_df) + 1))

    # Merge the predictions with the existing output DataFrame using 'user_id' and 'movie_id'
    merged_df = pd.merge(output_df, predictions_df, on=['user_id', 'movie_id'], how='left')

    # Save the updated DataFrame back to the CSV file
    merged_df.to_csv(existing_csv_file_path, index=False)

    print(f"Updated CSV file saved to {existing_csv_file_path}\n")