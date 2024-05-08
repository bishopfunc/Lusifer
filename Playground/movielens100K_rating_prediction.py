import openai
from openai import OpenAI
import pandas as pd
import json
from sklearn.metrics import mean_squared_error, precision_score, recall_score

# Set your OpenAI API key
KEY = "sk-proj-AycHzZMxqZscz8ltuD5iT3BlbkFJvJPLk9TbP9cMwDCZJd2w"

# path to the folder containing movielens data
Path = "D:/Canada/Danial/UoW/Dataset/MovieLens/100K/ml-100k"


# --------------------------------------------------------------
def get_last_10_ratings(user_id):
    """
    retrieve last 15 ratings according to the timestamp
    :param user_id:
    :return:
    """
    user_ratings = rating_df[rating_df['user_id'] == user_id].sort_values(by='timestamp', ascending=False).head(10)
    user_movies = movies_df[movies_df['movie_id'].isin(user_ratings['movie_id'])]
    return user_ratings.merge(user_movies, on='movie_id')


# --------------------------------------------------------------

def analyze_user_prompt(user_info, last_10_movies):
    """
    Generating the initial prompt to capture user's characteristics
    :param user_info: user information (age, gender, occupation)
    :param last_10_movies: dataframe
    :return:
    """
    # getting rating summary
    ratings_summary = '\n'.join(
        f"Movie: {row['movie_info']}\nRating: {row['rating']}" for _, row in last_10_movies.iterrows()
    )

    # Generating prompt
    prompt = f"""
Consider below information about the user:
User Info:
{user_info}

User's Last 15 Movies and Ratings:
{ratings_summary}

Analyze the user's characteristics based on their history and provide an indepth summary of the analysis as a text. 
include what type of movies the user enjoys or is not interested and provide important factors for the user. The 
output should be a JSON file. Below is an example of expected output:

"Summary": "User enjoys movies primarily in the genres of Comedy, Romance, and Drama. They have consistently rated 
movies in these genres highly (4.2 on average). On the other hand, the user seems less interested in movies in the 
genres of Animation, Sci-Fi, Action, and Thriller (2 on average). the user has a preference for character-driven 
narratives with emotional depth and relatable themes, and strong storyline."

"""

    return prompt


# --------------------------------------------------------------

def rate_new_movies_prompt(user_info, analysis, last_10_movies, test_movies):
    """
    Generate the proper prompt to ask the LLM to provide ratings for the recommendations
    :param user_info: user information (text)
    :param analysis: LLM's analysis based on user's background (text)
    :param test_movies: testset
    :return:
    """
    # test movie summaries
    movies_summary = '\n'.join(
        f"Movie ID: {row['movie_id']}\n{row['movie_info']}" for _, row in test_movies.iterrows()
    )

    prompt = f"""
Consider below information about a user
User Info:
{user_info}

User's last 10 movies:
{last_10_movies}

Analysis:
{analysis}

Based on the user information, user's last 10 movies, and user's characteristics from Analysis, rate the following 
movies (scale 1-5) on behalf of the user: {movies_summary}

Output should be a JSON file as example: movie_id(int): rating(int) 
"""
    return prompt


# --------------------------------------------------------------

def analyze_user(user_id):
    """
    Generates user's analysis
    :param user_id: int
    :return:
    """
    user_info = users_df[users_df['user_id'] == user_id]['user_info'].values[0]
    last_10_movies = get_last_10_ratings(user_id)
    prompt = analyze_user_prompt(user_info, last_10_movies)
    summary, tokens = get_llm_response(prompt, mode="summary")
    print("User Analysis:")
    print(summary)
    return summary, tokens, last_10_movies


# --------------------------------------------------------------
def rate_new_movies(user_id, analysis, last_10_movies):
    """
    Generate final output: simulated ratings
    """
    user_info = users_df[users_df['user_id'] == user_id]['user_info'].values[0]
    test_movies = rating_test_df[rating_test_df['user_id'] == user_id].merge(movies_df, on='movie_id')
    prompt = rate_new_movies_prompt(user_info, analysis, last_10_movies, test_movies)
    response, tokens = get_llm_response(prompt, mode="rating")
    return response, tokens


# --------------------------------------------------------------
# Function to compare the ratings and calculate accuracy metrics
def compare_ratings(user_id, llm_ratings):
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

    print("\nComparison:")
    print(comparison)

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

if __name__ == "__main__":
    # loading movielens dataset
    movies_df = pd.read_pickle("./Data/movies_enriched_dataset.pkl")
    movies_df = movies_df[["movie_id", "movie_info"]]

    # loading users_df
    users_df = pd.read_pickle("./Data/user_dataset.pkl")
    users_df = users_df[["user_id", "user_info"]]

    # Load the rating data
    rating_df = pd.read_csv(f'{Path}/u1.base', sep='\t', names=['user_id', 'movie_id', 'rating', 'timestamp'],
                            encoding='latin-1')
    rating_test_df = pd.read_csv(f'{Path}/u1.test', sep='\t', names=['user_id', 'movie_id', 'rating', 'timestamp'],
                                 encoding='latin-1')

    # Track token usage and evaluate results
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_cost = 0.0

    # Run the program for user 1
    user_id = 1
    summary, tokens_analysis, last_10_movies = analyze_user(user_id)

    # Add the "summary" column to users_df
    users_df['summary'] = users_df.apply(lambda row: summary if row['user_id'] == user_id else '', axis=1)

    total_prompt_tokens += tokens_analysis['prompt_tokens']
    total_completion_tokens += tokens_analysis['completion_tokens']

    llm_ratings, tokens_ratings = rate_new_movies(user_id, summary, last_10_movies)
    total_prompt_tokens += tokens_ratings['prompt_tokens']
    total_completion_tokens += tokens_ratings['completion_tokens']

    total_cost = (total_prompt_tokens + total_completion_tokens) * 0.0002  # Cost calculation per token

    print("\nLLM Ratings:")
    print(json.dumps(llm_ratings, indent=4))

    rmse, precision, recall = compare_ratings(user_id, llm_ratings)

    # Print token usage and cost estimation
    print("\nToken Usage and Cost:")
    print(f"Prompt Tokens: {total_prompt_tokens}")
    print(f"Completion Tokens: {total_completion_tokens}")
    print(f"Total Tokens: {total_prompt_tokens + total_completion_tokens}")
    print(f"Estimated Cost (USD): {total_cost:.5f}")
