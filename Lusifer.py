import openai
from openai import OpenAI
import pandas as pd
import json
from sklearn.metrics import root_mean_squared_error, precision_score, recall_score, accuracy_score
from tqdm import tqdm
import time
import os
import re

# Set your OpenAI API key
KEY = "sk-proj-AycHzZMxqZscz8ltuD5iT3BlbkFJvJPLk9TbP9cMwDCZJd2w"

# path to the folder containing movielens data
Path = "D:/Canada/Danial/UoW/Dataset/MovieLens/100K/ml-100k"


class Lusifer:
    """
    LLM-based User SImulated Feedback Environment for online Recommender systems:
    Lusifer can generate user summary behavior, updates the summary, and generate simulated ratings
    """

    def __init__(self, users_df, items_df, ratings_df):
        # loading data as pandas dataframes
        self.users_df = users_df
        self.items_df = items_df
        self.ratings_df = ratings_df

        self.api_key = None  # will be set by user
        self.model = None  # will be set by user

        self.user_feature = None  # will be set by user
        self.user_id = None
        self.item_feature = None  # will be set by user
        self.item_id = None
        self.timestamp = None

        # to trace the number of tokens and estimate the cost if needed
        self.temp_token_counter = 0
        self.total_tokens = {
            'prompt_tokens': 0,
            'completion_tokens': 0,
        }

        # prompts
        self.instructions = None

    # --------------------------------------------------------------
    def get_last_ratings(self, user_id, n=20):
        """
        Retrieve last N ratings according to the timestamp
        :param user_id:
        :param n: int or None (default is 20)
        :return: DataFrame
        """
        user_ratings = self.ratings_df[self.ratings_df[self.user_id] == user_id].sort_values(by=self.timestamp,
                                                                                             ascending=False)
        if n is not None:
            user_ratings = user_ratings.head(n)
        user_items = movies_df[self.items_df[self.item_id].isin(user_ratings[self.item_id])]
        return user_ratings.merge(user_items, on=self.item_id)

    # --------------------------------------------------------------
    # This function needs modification, users should be able to pass the prompt as an argument, but we are also using
    # variables to create the prompt dynamically.
    def analyze_user_prompt(self, user_info, last_n_items, n):
        """
        Generating the initial prompt to capture user's characteristics
        :param user_info: user information (age, gender, occupation)
        :param last_10_movies: dataframe
        :return:
        """
        # getting rating summary Below is the sample based on Movielens data
        ratings_summary = '\n'.join(
            f"Movie: {row['movie_info']}\nRating: {row['rating']}" for _, row in last_N_movies.iterrows()
        )

        # Generating prompt
        prompt = f"""
    Consider below information about the user:
    User Info:
    {user_info}
    
    User's Last {n} Movies and Ratings:
    {ratings_summary}
    
    Analyze the user's characteristics based on the history and provide an indepth summary of the analysis as a text. 
    include what type of movies the user enjoys or is not interested and provide important factors for the user. It 
    should be so clear that by reading the summary we could predict The user's potential rating based on the summary. 
    output should be a JSON file. Below is an example of expected output:
    
    {{"Summary": "User enjoys movies primarily in the genres of Comedy, Romance, and Drama. They have consistently rated 
    movies in these genres highly (4.2 on average). On the other hand, the user seems less interested in movies in the 
    genres of Animation, Sci-Fi, Action, and Thriller (2 on average). the user has a preference for character-driven 
    narratives with emotional depth and relatable themes, and strong storyline."}}
    """

        return prompt

    # --------------------------------------------------------------
    def rate_new_items_prompt(self, user_info, analysis, last_N_movies, test_movies):
        """
        Generate the proper prompt to ask the LLM to provide ratings for the recommendations
        :param user_info: user information (text)
        :param analysis: LLM's analysis based on user's background (text)
        :param test_movies: testset
        :return:
        """

        # recent movie summaries
        recent_movies_summary = '\n'.join(
            f"Movie: {row['movie_info']}\nRating: {row['rating']}" for _, row in last_N_movies.iterrows()
        )

        # test movie summaries
        movies_summary = '\n'.join(
            f"Movie ID: {row['movie_id']}\n{row['movie_info']}" for _, row in test_movies.iterrows()
        )

        prompt = f"""
    Consider below information about a user
    User Info:
    {user_info}
    
    User's most recent movies:
    {recent_movies_summary}
    
    Analysis:
    {analysis}
    
    Based on the user information, user's last 10 movies, and user's characteristics from Analysis, rate the following 
    movies (scale 1-5) on behalf of the user: {movies_summary}
    
    I want you to generate a JSON output containing movie ratings. The JSON format should be strictly as follows:
    
    {{
      "movie_id1": rating1,
      "movie_id2": rating2,
      ...
    }}
    
    Each key should be a movie_id (an integer), and each value should be a rating (an integer). Below is an example of 
    the ACCEPTED output:
    
    {{
      123: 4,
      456: 5
    }}
    
    Below is examples of the NOT ACCEPTED output:
    NOT ACCEPTED:
    {{
      "Movie ID": 123,
      "Rating": 4
    }}
    
    NOT ACCEPTED:
    {{
      'Movie ID: 33': 'Rating : 4'
    }}
    
    NOT ACCEPTED:
    {{
      'movie_id33': '4'
    }}
    
    Please ensure your response strictly follows the ACCEPTED format. Provide multiple movie ratings as needed.
    """
        return prompt

    # --------------------------------------------------------------
    def generate_summary(self, user_id, recent_items_to_consider=60, chunk_size=20):
        """
        Generates user's summary of behavior
        :param user_id: int
        :param recent_items_to_consider:
        :param chunk_size: int
        :return:
        """
        total_tokens = {
            'prompt_tokens': 0,
            'completion_tokens': 0,
        }

        user_info = self.users_df[self.users_df[self.user_id] == user_id][self.user_feature].values[0]
        last_n_items = self.get_last_ratings(user_id, n=recent_items_to_consider)

        # Get the first chunk of movies
        first_chunk = last_n_items[:chunk_size]
        prompt = self.generate_summary_prompt(user_info, first_chunk, n=chunk_size)
        summary, tokens = self.get_llm_response(prompt, mode="summary")

        # Process the remaining ratings in chunks and update the summary
        remaining_ratings = last_n_items[chunk_size:]  # Exclude the first chunk already analyzed
        for i in range(0, len(remaining_ratings), chunk_size):
            chunk = remaining_ratings[i:i + chunk_size]
            if not chunk.empty:
                prompt = self.update_summary_prompt(summary, user_info, chunk)
                summary, tokens_chunk = self.get_llm_response(prompt, mode="summary")
                self.total_tokens['prompt_tokens'] += tokens_chunk['prompt_tokens']
                self.total_tokens['completion_tokens'] += tokens_chunk['completion_tokens']

        return summary, tokens, last_n_items

    # --------------------------------------------------------------

    def update_summary_prompt(self, previous_summary, user_info, new_chunk):
        """
        Generate the prompt to update the summary with new movie ratings
        :param previous_summary: str
        :param new_chunk: DataFrame
        :return: str
        """
        ratings_summary = '\n'.join(
            f"Movie: {row['movie_info']}\nRating: {row['rating']}" for _, row in new_chunk.iterrows()
        )

        prompt = f"""
    Consider below information about the user:
    User Info:
    {user_info}
    
    Below is the Previous Summary information about user's characteristics based on their recent ratings:
    {previous_summary}
    
    Now, we have New Movie Ratings as below:
    {ratings_summary}
    
    Based on this comprehensive set of data, provide an in-depth summary of the user's movie preferences and 
    characteristics. Include details on the types of movies the user enjoys or is not interested in, and highlight 
    important factors that influence the user's preferences. The summary should be a coherent, stand-alone analysis that 
    integrates all the information without referring to updates or previous summaries. Consider adding more details than 
    previous summary given the new data. It should be so clear that by reading the summary we could predict The user's 
    potential rating based on the summary.The output should be a JSON file. The output should be a JSON file. Below is an 
    example of expected output:
    
    {{"Summary": "sample summary"}}
    """
        return prompt

    # --------------------------------------------------------------
    def rate_new_items(self, user_id, analysis, last_n_items, test_set, test_chunk_size=10):
        """
        Generate final output: simulated ratings
        """
        user_info = self.users_df[self.users_df[self.user_id] == user_id][self.user_feature].values[0]
        test_items = test_set.merge(self.items_df, on=self.item_feature)
        recent_items = last_n_items

        # Breaking test_set into chunks
        test_item_chunks = [test_items[i:i + test_chunk_size] for i in range(0, len(test_items), test_chunk_size)]

        aggregated_ratings = {}

        for chunk in test_item_chunks:
            prompt = self.rate_new_items_prompt(user_info, analysis, recent_items, chunk)
            response, tokens = self.get_llm_response(prompt, mode="rating")
            aggregated_ratings.update(response)
            self.total_tokens['prompt_tokens'] += tokens['prompt_tokens']
            self.total_tokens['completion_tokens'] += tokens['completion_tokens']

        return aggregated_ratings

    # --------------------------------------------------------------
    def get_llm_response(self, prompt, mode, max_retries=3):
        """
        sending the prompt to the LLM and get back the response
        """

        openai.api_key = self.api_key
        instructions = self.instructions
        client = OpenAI(api_key=KEY)

        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(
                    model=self.model,
                    response_format={"type": "json_object"},
                    messages=[
                        {"role": "system", "content": instructions},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=500,
                    n=1,
                    temperature=0.7
                )

                tokens = {
                    'prompt_tokens': response.usage.prompt_tokens,
                    'completion_tokens': response.usage.completion_tokens,
                    'total_tokens': response.usage.total_tokens
                }
                # print(f"Tokens used: {tokens}")

                try:
                    output = json.loads(response.choices[0].message.content)
                    if mode == "summary":
                        return output["Summary"], tokens
                    elif mode == "rating":
                        return output, tokens
                    else:
                        print(f"Invalid mode: {mode}")
                        return [], tokens

                except json.JSONDecodeError:
                    print(f"Invalid JSON from LLM on attempt {attempt + 1}. Retrying...")

            except openai.APIConnectionError as e:
                print("The server could not be reached")
                print(e.__cause__)  # an underlying Exception, likely raised within httpx.
            except openai.RateLimitError as e:
                print("A 429 status code was received; we should back off a bit.")
            except openai.APIStatusError as e:
                print("Another non-200-range status code was received")
                print(e.status_code)
                print(e.response)

        print("Max retries exceeded. Returning empty response.")
        return [], {}

    # --------------------------------------------------------------
    def load_data(self):
        """
        Serves as retry mechanism. In case of any interruption to the program, it should load the data from the last
        checkpoint :return:
        """

    # --------------------------------------------------------------
    def save_data(self, df, file_name):
        """
        Save the updated dataframes to files after each update to have checkpoint
         :return:
        """
        # Save the updated dataframes to files
        df.to_pickle(f'./Data/{file_name}.pkl')
        df.to_csv(f'./Data/{file_name}.csv')

    # --------------------------------------------------------------
    def filter_ratings(self, rating_test_df):
        """
        Make sure we have information about all the items in the test set
        :param rating_test_df: test set dataframe
        :return:
        """
        valid_item_ids = self.items_df[self.item_feature].unique()
        self.ratings_df = self.ratings_df[self.ratings_df[self.item_feature].isin(valid_item_ids)]
        rating_test_df = rating_test_df[rating_test_df['movie_id'].isin(valid_item_ids)]
        return rating_test_df

    # --------------------------------------------------------------
    def clean_key(self, key):
        """
        Handles unexpected LLM outputs: parsing key (item ids)
        :param key:
        :return:
        """
        # Use regex to extract numeric part of the key
        match = re.search(r'\d+', key)
        if match:
            return int(match.group(0))
        return None


    # --------------------------------------------------------------
    def clean_value(self, value):
        """
        Handles unexpected LLM outputs: parsing value (simulated rating)
        :param value:
        :return:
        """
        # Attempt to convert the value to an integer
        try:
            return int(value)
        except ValueError:
            return None

    # --------------------------------------------------------------
    def parse_llm_ratings(self, llm_ratings):
        """
        Parsing LLM output and handling common unexpected cases
        :param llm_ratings: dictionary of simualted ratings {item_id: rating}
        :return:
        """
        cleaned_ratings = {}
        for movie_id, rating in llm_ratings.items():
            clean_item_id = self.clean_key(movie_id)
            clean_rating = self.clean_value(rating)
            if clean_item_id is not None and clean_rating is not None:
                cleaned_ratings[clean_item_id] = clean_rating
        return cleaned_ratings


# --------------------------------------------------------------
if __name__ == "__main__":

    # loading movielens dataset
    movies_df, users_df, rating_df, rating_test_df = load_data()

    # Filtering out invalid movie_ids
    rating_df, rating_test_df = filter_ratings(movies_df, rating_df, rating_test_df)

    rmse_list, precision_list, recall_list, accuracy_list = [], [], [], []

    # user_ids = rating_test_df['user_id'].unique()
    user_ids = [1]

    # Track token usage and evaluate results
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_cost = 0.0

    temp_token_counter = 0

    # Generating user profile
    for user_id in tqdm(user_ids, desc="Processing users and generating summary profile"):
        if users_df.loc[users_df['user_id'] == user_id, 'summary'].any():
            continue

        else:
            summary, tokens_analysis, last_N_movies = analyze_user(user_id, recent_movies_to_consider=60)
            users_df.loc[users_df['user_id'] == user_id, 'summary'] = summary
            total_prompt_tokens += tokens_analysis['prompt_tokens']
            total_completion_tokens += tokens_analysis['completion_tokens']
            temp_token_counter += tokens_analysis['prompt_tokens'] + tokens_analysis['completion_tokens']

            # Check token limits
            if temp_token_counter > 55000:  # Using a safe margin
                print("Sleeping to respect the token limit...")
                # reset the token counter
                temp_token_counter = 0
                time.sleep(60)  # Sleep for a minute before making new requests

            # Saving summaries
            save_data(users_df, 'users_with_summary_df')

    temp_token_counter = 0
    # Generating the ratings
    for user_id in tqdm(user_ids, desc="Generating simulated ratings"):
        # isolating user's ratings in the test set
        user_ratings = rating_test_df[rating_test_df['user_id'] == user_id]

        # we might have some values from previous run
        missing_ratings = user_ratings[user_ratings['simulated_ratings'].isnull()]

        # getting the summary for the user
        summary = users_df.loc[users_df['user_id'] == user_id, 'summary'].values[0]

        # we will not run this part if we have all ratings for the user
        if not missing_ratings.empty:
            last_N_movies = get_last_ratings(user_id, n=10)

            llm_ratings, tokens_ratings = rate_new_movies(user_id, summary, last_N_movies, missing_ratings)
            total_prompt_tokens += tokens_ratings['prompt_tokens']
            total_completion_tokens += tokens_ratings['completion_tokens']
            temp_token_counter = tokens_ratings['prompt_tokens'] + tokens_ratings['completion_tokens']

            # Check token limits
            if temp_token_counter > 55000:  # Using a safe margin
                # reset counter
                temp_token_counter = 0
                print("Sleeping to respect the token limit...")
                time.sleep(60)  # Sleep for a minute before making new requests

            llm_ratings = parse_llm_ratings(llm_ratings)
            # llm_ratings = {int(movie_id): int(rating) for movie_id, rating in llm_ratings.items()}

            for movie_id, rating in llm_ratings.items():
                rating_test_df.loc[(rating_test_df['user_id'] == user_id) & (
                        rating_test_df['movie_id'] == movie_id), 'simulated_ratings'] = rating

            # rmse, precision, recall, accuracy = compare_ratings(user_id, llm_ratings, rating_test_df)
            # rmse_list.append(rmse)
            # precision_list.append(precision)
            # recall_list.append(recall)
            # accuracy_list.append(accuracy)

            save_data(rating_test_df, 'rating_test_df_test')

    # total_rmse = sum(rmse_list) / len(rmse_list)
    # total_precision = sum(precision_list) / len(precision_list)
    # total_recall = sum(recall_list) / len(recall_list)
    # total_accuracy = sum(accuracy_list) / len(accuracy_list)

    total_cost = ((total_prompt_tokens / 1000000) * 0.5) + (
            (total_completion_tokens / 1000000) * 1.5)  # Cost calculation

    print("\nToken Usage and Cost:")
    print(f"Prompt Tokens: {total_prompt_tokens}")
    print(f"Completion Tokens: {total_completion_tokens}")
    print(f"Total Tokens: {total_prompt_tokens + total_completion_tokens}")
    print(f"Estimated Cost (USD): {total_cost:.5f}")

    print("\nOverall Metrics:")

    evaluate_result(rating_test_df)
    # print(f"RMSE: {total_rmse:.4f}")
    # print(f"Precision: {total_precision:.4f}")
    # print(f"Recall: {total_recall:.4f}")
    # print(f"Accuracy: {total_accuracy:.4f}")
