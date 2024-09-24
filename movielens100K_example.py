import openai
from openai import OpenAI
import pandas as pd
import json
from sklearn.metrics import root_mean_squared_error, precision_score, recall_score, accuracy_score
from tqdm import tqdm
import time
import os
import re

# import Lusifer
from Lusifer import Lusifer

# Set your OpenAI API key
KEY = "OpenAI API"
# path to the folder containing movielens data
Path = "D:/Canada/Danial/UoW/Dataset/MovieLens/100K/ml-100k"


# --------------------------------------------------------------
def load_data():
    # Paths for the processed files
    processed_users_file = "./Samples/Data/users_with_summary_df.csv"
    processed_ratings_file = "./Samples/Data/rating_test_df_test.csv"

    # loading users dataframe
    if os.path.exists(processed_users_file):
        # Load the processed files if they exist
        users_df = pd.read_csv(processed_users_file)
        # users_df = pd.read_pickle(processed_users_file)

    else:
        users_df = pd.read_pickle("./Samples/Data/user_dataset.pkl")
        users_df = users_df[["user_id", "user_info"]]

    # loading ratings dataframe
    if os.path.exists(processed_ratings_file):
        rating_test_df = pd.read_csv(processed_ratings_file)

    else:
        rating_test_df = pd.read_csv(f'{Path}/u1.test', sep='\t', names=['user_id', 'movie_id', 'rating', 'timestamp'],
                                     encoding='latin-1')

    # loading ratings: Train set
    rating_df = pd.read_csv(f'{Path}/u1.base', sep='\t', names=['user_id', 'movie_id', 'rating', 'timestamp'],
                            encoding='latin-1')

    # Load movies
    movies_df = pd.read_pickle("./Samples/Data/movies_enriched_dataset.pkl")
    movies_df = movies_df[["movie_id", "movie_info"]]

    # Add new column to store simulated ratings if it doesn't exist
    if 'simulated_ratings' not in rating_test_df.columns:
        rating_test_df['simulated_ratings'] = None

    if 'summary' not in users_df.columns:
        users_df['summary'] = None

    return movies_df, users_df, rating_df, rating_test_df


# --------------------------------------------------------------
def compare_ratings(user_id, llm_ratings, rating_test_df):
    predicted_ratings = pd.DataFrame.from_dict(llm_ratings, orient='index', columns=['predicted']).astype(float)
    actual_ratings = rating_test_df[rating_test_df['user_id'] == user_id][['movie_id', 'rating']].set_index(
        'movie_id').astype(float)
    comparison = actual_ratings.join(predicted_ratings, how='inner').dropna()
    comparison.columns = ['actual', 'predicted']
    comparison['error'] = comparison['actual'] - comparison['predicted']

    rmse = root_mean_squared_error(comparison['actual'], comparison['predicted'])
    precision = precision_score(comparison['actual'], comparison['predicted'], average='micro')
    recall = recall_score(comparison['actual'], comparison['predicted'], average='micro')
    accuracy = accuracy_score(comparison['actual'], comparison['predicted'].round())

    return rmse, precision, recall, accuracy


# --------------------------------------------------------------
def evaluate_result(dataframe):
    dataframe = dataframe.dropna(subset=['simulated_ratings', 'rating'])

    dataframe['rating'] = dataframe['rating'].astype(int)
    dataframe['simulated_ratings'] = dataframe['simulated_ratings'].astype(int)

    # RMSE
    rmse = root_mean_squared_error(dataframe['rating'], dataframe['simulated_ratings'])

    # Exact match
    exact_matches = (dataframe['rating'] == dataframe['simulated_ratings'])
    exact_match_count = exact_matches.sum()
    exact_match_percentage = exact_match_count / len(dataframe) * 100

    # Off by 1 level
    off_by_1 = (dataframe['rating'] - dataframe['simulated_ratings']).abs() == 1
    off_by_1_count = off_by_1.sum()
    off_by_1_percentage = off_by_1_count / len(dataframe) * 100

    # Off by more than 1 level
    off_by_more_than_1 = (dataframe['rating'] - dataframe['simulated_ratings']).abs() > 1
    off_by_more_than_1_count = off_by_more_than_1.sum()
    off_by_more_than_1_percentage = off_by_more_than_1_count / len(dataframe) * 100

    # Weighted accuracy
    weighted_accuracy = (exact_matches * 1 + off_by_1 * 0.8).sum() / len(dataframe)

    # Output the results
    print(f"RMSE: {rmse}")
    print(f"Exact match count: {exact_match_count}")
    print(f"Exact match percentage: {exact_match_percentage:.2f}%")
    print(f"Off by 1 level count: {off_by_1_count}")
    print(f"Off by 1 level percentage: {off_by_1_percentage:.2f}%")
    print(f"Off by more than 1 level count: {off_by_more_than_1_count}")
    print(f"Off by more than 1 level percentage: {off_by_more_than_1_percentage:.2f}%")
    print(f"Weighted Accuracy: {weighted_accuracy:.2f}")


# --------------------------------------------------------------


if __name__ == "__main__":

    # loading movielens dataset
    movies_df, users_df, rating_df, rating_test_df = load_data()

    # create a Lusifer object
    lusifer = Lusifer(users_df=users_df,
                      items_df=movies_df,
                      ratings_df=rating_df)

    # set API connection
    lusifer.set_openai_connection(KEY, model="gpt-4o-mini")

    # set column names
    lusifer.set_column_names(user_feature="user_info",
                             item_feature="movie_info",
                             user_id="user_id",  # set by default
                             item_id="movie_id",
                             timestamp="timestamp",  # set by default
                             rating="rating")  # set by default

    # set LLM initial instruction
    instructions = """You are an AI assistant that receives users information and try to act like the user  by 
    analysing user's characteristics, profile and historical ratings inorder to provide new ratings for the recommended movies"""

    lusifer.set_llm_instruction(instructions)

    # you can set the prompts as below, or ignor this and use the default prompts
    # lusifer.set_prompts(prompt_summary, prompt_update_summary, prompt_simulate_rating)

    # you can set the path to store intermediate storing procedure. By default, they will be saved on Root.
    # lusifer.set_saving_path(self, path="")

    # Filtering out invalid movie_ids, make sure we have movie_info for every movie in the test set
    rating_test_df = lusifer.filter_ratings(rating_test_df)
    rating_df = lusifer.filter_ratings(rating_df)


    user_ids = rating_test_df['user_id'].unique()
    # user_ids = [1]

    # Track token usage and evaluate results
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_cost = 0.0

    temp_token_counter = 0

    ## PHASE 1: Generating Summary of User's Behavior

    # Generating user profile
    for user_id in tqdm(user_ids, desc="Processing users and generating summary profile"):
        if users_df.loc[users_df['user_id'] == user_id, 'summary'].any():
            continue

        else:
            # generating summary for the user using Lusifer
            summary, tokens_analysis, last_N_movies = lusifer.generate_summary(user_id, recent_movies_to_consider=60)
            users_df.loc[users_df['user_id'] == user_id, 'summary'] = summary

            # Check token limits
            if temp_token_counter > 55000:  # Using a safe margin
                print("Sleeping to respect the token limit...")
                # reset the token counter
                temp_token_counter = 0
                time.sleep(60)  # Sleep for a minute before making new requests

            # Saving summaries
            lusifer.save_data(users_df, 'users_with_summary_df')

    temp_token_counter = 0

    ## PHASE 2: Generating Simulated ratings
    for user_id in tqdm(user_ids, desc="Generating simulated ratings"):
        # isolating user's ratings in the test set
        user_ratings = rating_test_df[rating_test_df['user_id'] == user_id]

        # we might have some values from previous run
        missing_ratings = user_ratings[user_ratings['simulated_ratings'].isnull()]

        # getting the summary for the user
        summary = users_df.loc[users_df['user_id'] == user_id, 'summary'].values[0]

        # we will not run this part if we have all ratings for the user
        if not missing_ratings.empty:
            last_N_movies = lusifer.get_last_ratings(user_id, n=10)

            # generate rating
            llm_ratings, tokens_ratings = lusifer.rate_new_items(summary, last_N_movies, missing_ratings)
            total_prompt_tokens += tokens_ratings['prompt_tokens']
            total_completion_tokens += tokens_ratings['completion_tokens']
            temp_token_counter = tokens_ratings['prompt_tokens'] + tokens_ratings['completion_tokens']

            # Check token limits
            if temp_token_counter > 55000:  # Using a safe margin
                # reset counter
                temp_token_counter = 0
                print("Sleeping to respect the token limit...")
                time.sleep(60)  # Sleep for a minute before making new requests

            # parsing the output JSON : Error handling
            llm_ratings = lusifer.parse_llm_ratings(llm_ratings)

            # Assigning the ratings to the movies
            for movie_id, rating in llm_ratings.items():
                rating_test_df.loc[(rating_test_df['user_id'] == user_id) & (
                        rating_test_df['movie_id'] == movie_id), 'simulated_ratings'] = rating


            lusifer.save_data(rating_test_df, 'rating_test_df_test')



    total_cost = ((total_prompt_tokens / 1000000) * 0.5) + (
            (total_completion_tokens / 1000000) * 1.5)  # Cost calculation estimation

    print("\nToken Usage and Cost:")
    print(f"Prompt Tokens: {total_prompt_tokens}")
    print(f"Completion Tokens: {total_completion_tokens}")
    print(f"Total Tokens: {total_prompt_tokens + total_completion_tokens}")
    print(f"Estimated Cost (USD): {total_cost:.5f}")

    print("\nOverall Metrics:")

    evaluate_result(rating_test_df)

