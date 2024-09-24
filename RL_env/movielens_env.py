import gym
from gym import spaces
import numpy as np
import json
from Lusifer import Lusifer
import pandas as pd
import time


class MovieRecommenderEnv(gym.Env):
    def __init__(self, Lusifer):
        """
        initializing the environment
        :param Lusifer: a Lusifer object
        """
        self.lusifer = Lusifer

        self.users_df = Lusifer.users_df  # user dataframe
        self.movies_df = Lusifer.items_df  # item dataframe (movies)
        self.interactions_df = Lusifer.ratings_df  # ratings dataframe

        self.num_users = len(self.users_df)
        self.num_movies = len(self.movies_df)

        # vector embedding (BERT)-> 768
        self.embedding_size = 768

        # Action space: Indices of movies (0 to num_movies-1)
        self.action_space = spaces.MultiDiscrete([self.num_movies])

        # Observation space: User profile embedding
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.embedding_size,), dtype=np.float32)

        self.current_user = None
        self.current_state = None

        # Tracking token used to respect limit rates
        self.temp_token_counter = 0
        self.total_tokens = {
            'prompt_tokens': 0,
            'completion_tokens': 0,
        }
        self.token_clock = 0  # resets every minute

    def reset(self):
        # Randomly select a user to start with
        self.current_user = np.random.choice(list(self.user_profiles.keys()))
        self.current_state = self.user_profiles[self.current_user]
        return self.current_state

    def step(self, action):
        """
        steps that the agent takes in the environment
        :param action:
        :return:
        """
        """
        Recommend movies.
        Get ratings.
        Update user profiles.
        Compute rewards.
        Return the next state, reward, and done flag.
        """

        # Action is a list of movie_ids to recommend, we need to find the recommended movie using "movie_id" in self.movies_df

        # Get the ratings from the user for the recommended movies
        ratings = self.get_ratings(user=self.current_user, recommended_items=action)

        # updating ratings dataframe with new ratings


        # Update user profile based on the new ratings using LUSIFER


        # get vector embeddings of the new user_profile


        # update current_state


        # compute the reward


        self.current_state = self.update_user_profile(self.current_user, recommended_movies, ratings)


        # Check if the episode is done (optional, based on some criteria)
        done = False

        return self.current_state, reward, done, {}

    def render(self, mode='human', close=False):
        pass

    def get_ratings(self, user, recommended_items):
        """
        Sending recommendations through Lusifer object for the current user, and getting the ratings for the items
        :return:
        """

        user_id = user['user_id'].values[0]
        user_profile = user['user_profile'].values[0]

        # fetching last n movies (dataframe) for the current user to include it in the prompt
        last_N_movies = self.lusifer.get_last_ratings(user_id, n=5)

        # generate rating
        llm_ratings = self.lusifer.rate_new_items(user_profile=user_profile,
                                                  last_n_items=last_N_movies,
                                                  test_set=recommended_items)

        # parsing the output JSON: Error handling
        llm_ratings = self.lusifer.parse_llm_ratings(llm_ratings)

        # llm_ratings would be a dictionary where the keys are movie_ids and the ratings are the values
        return llm_ratings


# Example usage:
# user_profiles = {user_id: initial_user_embedding}
# movie_profiles = [movie_embedding_1, movie_embedding_2, ..., movie_embedding_N]
# env = MovieRecommenderEnv(user_profiles, movie_profiles, get_embedding, update_user_profile, get_ratings)
# obs = env.reset()
# action = env.action_space.sample()
# next_state, reward, done, _ = env.step(action)


def load_config_(file_path):
    """
    Loads configuration JSON files from the local space. (mostly for loading the Snowflake connection parameters)
    :param file_path: local path to the JSON file
    :return: JSON file
    """
    with open(file_path, 'r') as file:
        return json.load(file)


if __name__ == "__main__":
    # path to sample data
    path = "sample_data/7_10_days_no_interaction.csv"

    # loading sample data
    users = pd.read_csv(path)

    config_files = '../config_files/config.json'
    config = load_config_(config_files)
