import pandas as pd

from movielens100K_example import evaluate_result

if __name__ == "__main__":
    rating_test_df = pd.read_csv("rating_test_df_test.csv")
    print(rating_test_df)
    evaluate_result(rating_test_df)
