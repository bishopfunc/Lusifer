import openai
from openai import OpenAI
import json
import re


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
        self.rating = None

        # to trace the number of tokens and estimate the cost if needed
        self.temp_token_counter = 0
        self.total_tokens = {
            'prompt_tokens': 0,
            'completion_tokens': 0,
        }

        # prompts
        self.instructions = None
        self.prompt_summary = None
        self.prompt_update_summary = None
        self.prompt_simulate_rating = None

        # saving path
        self.saving_path = None

    # --------------------------------------------------------------
    def set_openai_connection(self, api_key, model):
        """
        Setting openai connection
        :param api_key: openai Key
        :param model: LLM model from openai API
        :return:
        """
        self.api_key = api_key
        self.model = model

    # --------------------------------------------------------------
    def set_column_names(self, user_feature, item_feature,
                         user_id="user_id",
                         item_id="item_id",
                         timestamp="timestamp",
                         rating="rating"):
        """
        Setting necessary column names
        :param user_feature: user feature column
        :param item_feature: item feature column
        :param user_id: user_id column
        :param item_id: item_id column
        :param timestamp: timestamp column
        :param rating: rating column
        :return:
        """

        self.user_feature = user_feature  # will be set by user
        self.item_feature = item_feature  # will be set by user

        self.user_id = user_id
        self.item_id = item_id

        self.timestamp = timestamp
        self.rating = rating

    # --------------------------------------------------------------
    def set_llm_instruction(self, instructions):
        """
        Set initial instruction of the LLM model
        :param instructions:
        :return:
        """

        self.instructions = instructions

    # --------------------------------------------------------------
    def set_prompts(self, prompt_summary, prompt_update_summary, prompt_simulate_rating):
        """
        Set prompts for Lusifer
        :param prompt_summary: prompt to generate the first summary
        :param prompt_update_summary: prompt to update the summary
        :param prompt_simulate_rating:
        :return:
        """
        self.prompt_summary = None
        self.prompt_update_summary = None
        self.prompt_simulate_rating = None

    # --------------------------------------------------------------

    def set_saving_path(self, path=""):
        """
        Setting openai connection
        :param path: path to the folder you want to store the intermediate progress of Lusifer
        :return:
        """
        self.saving_path = path

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
    def generate_summary_prompt(self, user_info, last_n_items, n):
        """
        Generating the initial prompt to capture user's characteristics
        :param user_info: user information
        :param last_n_items: dataframe containing last n items
        :param n: last n items (integer)
        :return:
        """
        if self.prompt_summary == None:
            self.prompt_summary = """Analyze the user's characteristics based on the history and provide an indepth 
            summary of the analysis as a text. include what type of items the user enjoys or is not interested and 
            provide important factors for the user. It should be so clear that by reading the summary we could 
            predict The user's potential rating based on the summary. """

        # getting rating summary Below is the sample based on Movielens data
        ratings_summary = '\n'.join(
            f"Item: {row[self.item_feature]}\nRating: {row[self.rating]}" for _, row in last_n_items.iterrows()
        )

        # Generating prompt
        prompt = f"""
    Consider below information about the user:
    User Info:
    {user_info}
    
    User's Last {n} items and Ratings:
    {ratings_summary}
        
    {self.prompt_summary}
    
    output should be a JSON file. Below is an example of expected output:
    
    {{"Summary": "sample summary "}}
    """

        return prompt

    # --------------------------------------------------------------
    def rate_new_items_prompt(self, user_info, summary, last_n_movies, test_set):
        """
        Generate the proper prompt to ask the LLM to provide ratings for the recommendations
        :param user_info: user information (text)
        :param analysis: LLM's summary of user's behavior based on user's background (text)
        :param test_movies: testset
        :return:
        """
        if self.prompt_simulate_rating == None:
            self.prompt_simulate_rating = """Based on the user information, user's last 10 items, and user's 
            characteristics from Analysis, rate the following items (scale 1-5) on behalf of the user:"""

        # recent items summaries
        recent_items_summary = '\n'.join(
            f"Item: {row[self.item_feature]}\nRating: {row[self.rating]}" for _, row in last_n_movies.iterrows()
        )

        # test items summaries
        items_summary = '\n'.join(
            f"Item ID: {row[self.item_id]}\n{row[self.item_feature]}" for _, row in test_set.iterrows()
        )

        prompt = f"""
    Consider below information about a user
    User Info:
    {user_info}
    
    User's most recent items:
    {recent_items_summary}
    
    Summary of User's behavior:
    {summary}
    
    {self.prompt_simulate_rating}
    
    {items_summary}
    
    I want you to generate a JSON output containing item ratings. The JSON format should be strictly as follows:
    
    {{
      "item_id1": rating1,
      "item_id2": rating2,
      ...
    }}
    
    Each key should be a item_id (an integer), and each value should be a rating (an integer). Below is an example of 
    the ACCEPTED output:
    
    {{
      123: 4,
      456: 5
    }}
    
    Below is examples of the NOT ACCEPTED output:
    NOT ACCEPTED:
    {{
      "Item ID": 123,
      "Rating": 4
    }}
    
    NOT ACCEPTED:
    {{
      'Item ID: 33': 'Rating : 4'
    }}
    
    NOT ACCEPTED:
    {{
      'item_id33': '4'
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
        Generate the prompt to update the summary with new item ratings
        :param previous_summary: str
        :param new_chunk: DataFrame
        :return: str
        """
        if self.prompt_update_summary == None:
            self.prompt_update_summary = """Based on this comprehensive set of data, provide an in-depth summary of 
            the user's movie preferences and characteristics. Include details on the types of movies the user enjoys 
            or is not interested in, and highlight important factors that influence the user's preferences. The 
            summary should be a coherent, stand-alone analysis that integrates all the information without referring 
            to updates or previous summaries. Consider adding more details than previous summary given the new data. 
            It should be so clear that by reading the summary we could predict The user's potential rating based on 
            the summary."""

        ratings_summary = '\n'.join(
            f"Item: {row[self.item_feature]}\nRating: {row[self.rating]}" for _, row in new_chunk.iterrows()
        )

        prompt = f"""
    Consider below information about the user:
    User Info:
    {user_info}
    
    Below is the Previous Summary information about user's characteristics based on their recent ratings:
    {previous_summary}
    
    Now, we have New items Ratings as below:
    {ratings_summary}
    
    {self.prompt_update_summary}
    
    The output should be a JSON file. The output should be a JSON file. Below is an example of expected output:
    
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
        client = OpenAI(api_key=self.api_key)

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
    def save_data(self, df, file_name):
        """
        Save the updated dataframes to files after each update to have checkpoint
         :return:
        """
        # Save the updated dataframes to files
        df.to_pickle(f'{self.saving_path}{file_name}.pkl')
        df.to_csv(f'{self.saving_path}{file_name}.csv')

    # --------------------------------------------------------------
    def filter_ratings(self, rating_test_df):
        """
        Make sure we have information about all the items in the test set
        :param rating_test_df: dataframe
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
