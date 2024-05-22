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

{
  "movie_id1": rating1,
  "movie_id2": rating2,
  ...
}

Each key should be a movie_id (an integer), and each value should be a rating (an integer). Below is an example of 
the ACCEPTED output:


{
  123: 4,
  456: 5
}

Below is examples of the NOT ACCEPTED output:
NOT ACCEPTED:

{
  "Movie ID": 123,
  "Rating": 4
}

NOT ACCEPTED:

{
  'Movie ID: 33': 'Rating : 4'
}

NOT ACCEPTED:

{
  'movie_id33': '4'
}

Please ensure your response strictly follows the ACCEPTED format. Provide multiple movie ratings as needed.