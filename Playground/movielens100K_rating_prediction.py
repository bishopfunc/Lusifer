import openai
from openai import OpenAI
import pandas as pd
import json
from sklearn.metrics import mean_squared_error, precision_score, recall_score, accuracy_score
from tqdm import tqdm

# Set your OpenAI API key
KEY = "sk-proj-AycHzZMxqZscz8ltuD5iT3BlbkFJvJPLk9TbP9cMwDCZJd2w"

# path to the folder containing movielens data
Path = "D:/Canada/Danial/UoW/Dataset/MovieLens/100K/ml-100k"


# --------------------------------------------------------------
def get_last_ratings(user_id, n=20):
    """
    Retrieve last N ratings according to the timestamp
    :param user_id:
    :param n: int or None
    :return: DataFrame
    """
    user_ratings = rating_df[rating_df['user_id'] == user_id].sort_values(by='timestamp', ascending=False)
    if n is not None:
        user_ratings = user_ratings.head(n)
    user_movies = movies_df[movies_df['movie_id'].isin(user_ratings['movie_id'])]
    return user_ratings.merge(user_movies, on='movie_id')


# --------------------------------------------------------------

def analyze_user_prompt(user_info, last_N_movies, n):
    """
    Generating the initial prompt to capture user's characteristics
    :param user_info: user information (age, gender, occupation)
    :param last_10_movies: dataframe
    :return:
    """
    # getting rating summary
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

"Summary": "User enjoys movies primarily in the genres of Comedy, Romance, and Drama. They have consistently rated 
movies in these genres highly (4.2 on average). On the other hand, the user seems less interested in movies in the 
genres of Animation, Sci-Fi, Action, and Thriller (2 on average). the user has a preference for character-driven 
narratives with emotional depth and relatable themes, and strong storyline."
"""

    return prompt


# --------------------------------------------------------------

def rate_new_movies_prompt(user_info, analysis, last_N_movies, test_movies):
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

Output should be a JSON file where the keys are movie_id and values are ratings(1-5), see below example 
123: 4 (in this example, 123 is movie_id and 4 is the rating)


"""
    return prompt


def analyze_user(user_id, recent_movies_to_consider=60, chunk_size=20):
    """
    Generates user's analysis
    :param user_id: int
    :param recent_movies_to_consider: int
    :param chunk_size: int
    :return:
    """
    total_tokens = {
        'prompt_tokens': 0,
        'completion_tokens': 0,
    }

    user_info = users_df[users_df['user_id'] == user_id]['user_info'].values[0]
    last_N_movies = get_last_ratings(user_id, n=recent_movies_to_consider)

    # Get the first chunk of movies
    first_chunk = last_N_movies[:chunk_size]
    prompt = analyze_user_prompt(user_info, first_chunk, n=chunk_size)
    summary, tokens = get_llm_response(prompt, mode="summary")

    # # Print initial analysis
    # print("User Analysis (Initial Chunk):")
    # print(summary)

    # Process the remaining ratings in chunks and update the summary
    remaining_ratings = last_N_movies[chunk_size:]  # Exclude the first chunk already analyzed
    for i in range(0, len(remaining_ratings), chunk_size):
        chunk = remaining_ratings[i:i + chunk_size]
        if not chunk.empty:
            prompt = update_summary_prompt(summary, user_info, chunk)
            summary, tokens_chunk = get_llm_response(prompt, mode="summary")
            tokens['prompt_tokens'] += tokens_chunk['prompt_tokens']
            tokens['completion_tokens'] += tokens_chunk['completion_tokens']

            # # Print updated analysis
            # print(f"User Analysis (Chunk {i // chunk_size + 1}):")
            # print(summary)

    return summary, tokens, last_N_movies

# --------------------------------------------------------------

def update_summary_prompt(previous_summary, user_info, new_chunk):
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
previous summary given the new data. The output should be a JSON file. The output should be a JSON file. Below is an 
example of expected output:

"Summary": "sample summary"
"""
    return prompt


# --------------------------------------------------------------
def rate_new_movies(user_id, analysis, last_N_movies):
    """
    Generate final output: simulated ratings
    """
    user_info = users_df[users_df['user_id'] == user_id]['user_info'].values[0]
    test_movies = rating_test_df[rating_test_df['user_id'] == user_id].merge(movies_df, on='movie_id')

    recent_movies = get_last_ratings(user_id, n=10)

    # Breaking test_movies into chunks of 10
    chunk_size = 10
    test_movie_chunks = [test_movies[i:i + chunk_size] for i in range(0, len(test_movies), chunk_size)]

    aggregated_ratings = {}
    total_tokens = {
        'prompt_tokens': 0,
        'completion_tokens': 0,
    }

    for chunk in test_movie_chunks:
        prompt = rate_new_movies_prompt(user_info, analysis, recent_movies, chunk)
        response, tokens = get_llm_response(prompt, mode="rating")
        aggregated_ratings.update(response)
        total_tokens['prompt_tokens'] += tokens['prompt_tokens']
        total_tokens['completion_tokens'] += tokens['completion_tokens']

    return aggregated_ratings, total_tokens

# --------------------------------------------------------------
# Function to compare the ratings and calculate accuracy metrics
def compare_ratings_individual_user(user_id, llm_ratings):
    """
    Function to compare the ratings and calculate accuracy
    :param user_id:
    :param llm_ratings:
    :return:
    """
    # Create a DataFrame from the provided llm_ratings dictionary
    predicted_ratings = pd.DataFrame.from_dict(llm_ratings, orient='index', columns=['predicted']).astype(float)

    # Extract actual ratings for the specified user from rating_test_df
    actual_ratings = rating_test_df[rating_test_df['user_id'] == user_id][['movie_id', 'rating']].set_index(
        'movie_id').astype(float)

    # Join the actual and predicted ratings DataFrames
    comparison = actual_ratings.join(predicted_ratings, how='inner').dropna()
    comparison.columns = ['actual', 'predicted']
    comparison['error'] = comparison['actual'] - comparison['predicted']

    # print("\nComparison:")
    # print(comparison)

    # Calculate metrics
    rmse = mean_squared_error(comparison['actual'], comparison['predicted'], squared=False)
    precision = precision_score(comparison['actual'], comparison['predicted'], average='micro')
    recall = recall_score(comparison['actual'], comparison['predicted'], average='micro')


    print("\nMetrics:")
    print(f"RMSE: {rmse:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")

    return rmse, precision, recall


# --------------------------------------------------------------
def get_llm_response(prompt, mode):
    """
    sending the prompt to the LLM and get back the response
    """

    openai.api_key = KEY

    instructions = """You are an AI assistant that receives users information and try to act like the user  by 
    analysing user's characteristics inorder to provide ratings for the recommendations"""

    client = OpenAI(api_key=KEY)

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo-0125",
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": instructions},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500,
            n=1,
            temperature=0.7
        )

        tokens = response.usage

        tokens = {
            'prompt_tokens': response.usage.prompt_tokens,
            'completion_tokens': response.usage.completion_tokens,
            'total_tokens': response.usage.total_tokens
        }
        print(f"Tokens used: {tokens}")


        try:
            output = json.loads(response.choices[0].message.content)
            if mode == "summary":
                return output["Summary"], tokens
            elif mode == "rating":
                return output, tokens
            else:
                print(f"invalid mode: {mode}")

        except json.JSONDecodeError:
            print("Invalid JSON from LLM. Adjust the prompt or review the response.")
            return [], tokens

        return response.choices[0].message.content.strip(), tokens

    except openai.APIConnectionError as e:
        print("The server could not be reached")
        print(e.__cause__)  # an underlying Exception, likely raised within httpx.
    except openai.RateLimitError as e:
        print("A 429 status code was received; we should back off a bit.")
    except openai.APIStatusError as e:
        print("Another non-200-range status code was received")
        print(e.status_code)
        print(e.response)

# --------------------------------------------------------------
def load_data():
    # Load the datasets
    movies_df = pd.read_pickle("./Data/movies_enriched_dataset.pkl")
    movies_df = movies_df[["movie_id", "movie_info"]]

    users_df = pd.read_pickle("./Data/user_dataset.pkl")
    users_df = users_df[["user_id", "user_info"]]

    rating_df = pd.read_csv(f'{Path}/u1.base', sep='\t', names=['user_id', 'movie_id', 'rating', 'timestamp'], encoding='latin-1')
    rating_test_df = pd.read_csv(f'{Path}/u1.test', sep='\t', names=['user_id', 'movie_id', 'rating', 'timestamp'], encoding='latin-1')

    # Add new column to store simulated ratings if it doesn't exist
    if 'simulated_ratings' not in rating_test_df.columns:
        rating_test_df['simulated_ratings'] = None

    return movies_df, users_df, rating_df, rating_test_df

# --------------------------------------------------------------
def save_data(users_df, rating_test_df):
    # Save the updated dataframes to files
    rating_test_df.to_csv('./Data/rating_test_with_simulated_ratings.pkl')
    users_df.to_pickle('./Data/user_dataset_with_summaries.pkl')

# --------------------------------------------------------------
def compare_ratings(user_id, llm_ratings, rating_test_df):

    predicted_ratings = pd.DataFrame.from_dict(llm_ratings, orient='index', columns=['predicted']).astype(float)
    actual_ratings = rating_test_df[rating_test_df['user_id'] == user_id][['movie_id', 'rating']].set_index(
        'movie_id').astype(float)
    comparison = actual_ratings.join(predicted_ratings, how='inner').dropna()
    comparison.columns = ['actual', 'predicted']
    comparison['error'] = comparison['actual'] - comparison['predicted']

    rmse = mean_squared_error(comparison['actual'], comparison['predicted'], squared=False)
    precision = precision_score(comparison['actual'], comparison['predicted'], average='micro')
    recall = recall_score(comparison['actual'], comparison['predicted'], average='micro')
    accuracy = accuracy_score(comparison['actual'], comparison['predicted'].round())

    return rmse, precision, recall, accuracy

# --------------------------------------------------------------
if __name__ == "__main__":

    # loading movielens dataset
    movies_df, users_df, rating_df, rating_test_df = load_data()

    rmse_list, precision_list, recall_list, accuracy_list = [], [], [], []

    # user_ids = rating_test_df['user_id'].unique()
    # test
    user_ids = [2, 3, 4]

    # Track token usage and evaluate results
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_cost = 0.0

    # generating user profile summary
    for user_id in tqdm(user_ids, desc="Processing users and generating summary profile"):
        if users_df.loc[users_df['user_id'] == user_id, 'summary'].any():
            continue

        else:
            summary, tokens_analysis, last_N_movies = analyze_user(user_id, recent_movies_to_consider=60)
            users_df.loc[users_df['user_id'] == user_id, 'summary'] = summary
            total_prompt_tokens += tokens_analysis['prompt_tokens']
            total_completion_tokens += tokens_analysis['completion_tokens']

    for user_id in tqdm(user_ids, desc="generating simulated ratings"):

            llm_ratings, tokens_ratings = rate_new_movies(user_id, summary, last_N_movies)
            total_prompt_tokens += tokens_ratings['prompt_tokens']
            total_completion_tokens += tokens_ratings['completion_tokens']

            llm_ratings = {int(movie_id): int(rating) for movie_id, rating in llm_ratings.items()}

            for movie_id, rating in llm_ratings.items():
                rating_test_df.loc[(rating_test_df['user_id'] == user_id) & (
                            rating_test_df['movie_id'] == movie_id), 'simulated_ratings'] = rating

            rmse, precision, recall, accuracy = compare_ratings(user_id, llm_ratings, rating_test_df)

            rmse_list.append(rmse)
            precision_list.append(precision)
            recall_list.append(recall)
            accuracy_list.append(accuracy)

            save_data(users_df, rating_test_df)

    total_rmse = sum(rmse_list) / len(rmse_list)
    total_precision = sum(precision_list) / len(precision_list)
    total_recall = sum(recall_list) / len(recall_list)
    total_accuracy = sum(accuracy_list) / len(accuracy_list)

    total_cost = ((total_prompt_tokens / 1000000) * 0.5) + (
                (total_completion_tokens / 1000000) * 1.5)  # Cost calculation

    print("\nToken Usage and Cost:")
    print(f"Prompt Tokens: {total_prompt_tokens}")
    print(f"Completion Tokens: {total_completion_tokens}")
    print(f"Total Tokens: {total_prompt_tokens + total_completion_tokens}")
    print(f"Estimated Cost (USD): {total_cost:.5f}")

    print("\nOverall Metrics:")
    print(f"RMSE: {total_rmse:.4f}")
    print(f"Precision: {total_precision:.4f}")
    print(f"Recall: {total_recall:.4f}")
    print(f"Accuracy: {total_accuracy:.4f}")
