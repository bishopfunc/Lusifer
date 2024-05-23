# Lusifer: LLM-based User Simulated Feedback Environment for Online Recommender Systems

Lusifer (LLM-based User SImulated Feedback Environment for online Recommender systems) is a platform designed to simulate user behavior and generate feedback for recommender systems using Large Language Models (LLMs). This environment helps researchers evaluate and improve their recommender systems by providing a scalable and adaptable framework for user simulation.

For more information, read the paper:  
**[Lusifer: LLM-based User Simulated Feedback Environment for Online Recommender Systems](https://github.com/danialebrat/Lusifer)**

## Getting Started

### Set OpenAI Connection

```python
from lusifer import Lusifer 
lusifer = Lusifer(users_df, items_df, ratings_df) 
lusifer.set_openai_connection(api_key='your_openai_api_key', model='gpt-3.5-turbo') 
```

### Set LLM Instructions and Prompts

```python
lusifer.set_llm_instruction(instructions="your_llm_instructions") 
lusifer.set_prompts(
    prompt_summary="your_prompt_to_generate_summary", 
    prompt_update_summary="your_prompt_to_update_summary", 
    prompt_simulate_rating="your_prompt_to_simulate_rating" 
)
```


### set column names

```python
lusifer.set_column_names(user_feature="user_info",
                         item_feature="movie_info",
                         user_id="user_id",  # set by default
                         item_id="movie_id",
                         timestamp="timestamp",  # set by default
                         rating="rating")  # set by default

    
# you can set the prompts as below, or ignor this and use the default prompts
lusifer.set_prompts(prompt_summary, prompt_update_summary, prompt_simulate_rating)

# you can set the path to store intermediate storing procedure. By default, they will be saved on Root.
lusifer.set_saving_path(self, path="")
```


## Run the Sample Experiment
A sample experiment on the MovieLens dataset is available: [here](https://github.com/danialebrat/Lusifer/blob/master/movielens100K_example.py)

## Access Sample Prompts, Datasets, and Outputs
Explore sample prompts, datasets, and outputs [here](https://github.com/danialebrat/Lusifer/blob/master/movielens100K_example.py](https://github.com/danialebrat/Lusifer/tree/master/Samples).

Contributions
This is the first version of Lusifer. We welcome contributions and suggestions to improve and expand the capabilities of this tool.

## References
Paper: Lusifer: LLM-based User Simulated Feedback Environment for Online Recommender Systems.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Contact
For any inquiries or further information, please contact Danial Ebrat.
