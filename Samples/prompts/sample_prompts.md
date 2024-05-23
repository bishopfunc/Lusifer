## Sample prompt for generating the summary of user's behavior (first time)
"""

Analyze the user's characteristics based on the history and provide an indepth 
summary of the analysis as a text. include what type of items the user enjoys or is not interested and 
provide important factors for the user. It should be so clear that by reading the summary we could 
predict The user's potential rating based on the summary.

"""



## Sample prompt for updating the summary of user's behavior

"""

Based on this comprehensive set of data, provide an in-depth summary of 
the user's movie preferences and characteristics. Include details on the types of movies the user enjoys 
or is not interested in, and highlight important factors that influence the user's preferences. The 
summary should be a coherent, stand-alone analysis that integrates all the information without referring 
to updates or previous summaries. Consider adding more details than previous summary given the new data. 
It should be so clear that by reading the summary, we could predict The user's potential rating based on 
the summary.

"""




## Sample prompt for generating simulated ratings:

"""

Based on the user information, user's last 10 movies, and user's characteristics from Analysis, rate the following 
movies (scale 1-5) on behalf of the user: 

"""



Note that the Lusifer will take care of generating other parts of the final prompt including but not limited to:
- adding last_n_interactions (item's information and user's ratings)
- adding a summary of user's behavior
- adding item's information if needed
- adding instruction to guide the LLM for the proper output


