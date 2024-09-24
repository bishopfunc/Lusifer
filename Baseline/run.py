from Recommender import test_recommender
from surprise import KNNBasic, SVD

# Maximum number of entries from a user that will be used during training, the most recent ones will be used
max_entries_per_user = 60  # If zero or less it will allow an unlimited number to be used in training

# Loop from 1-5 for the 5 Movielens 100k test sets
for index in range(1, 6):
    test_recommender(KNNBasic(sim_options={'name': 'cosine', 'user_based': True}),
                     "Movielens100K/ml-100k",
                     "Movielens100K",
                     index,
                     max_entries_per_user,
                     "knn_predicted_ratings")

    test_recommender(SVD(),
                     "Movielens100K/ml-100k",
                     "Movielens100K",
                     index,
                     max_entries_per_user,
                     "matrix_factorization_predicted_ratings")

print("Movielens 100k done\n\n")


# Loop from 1-5 for the 5 Movielens 1m test sets
for index in range(1, 6):
    test_recommender(KNNBasic(sim_options={'name': 'cosine', 'user_based': True}),
                     "Movielens1m/ml-1m",
                     "Movielens1m",
                     index,
                     max_entries_per_user,
                     "knn_predicted_ratings")

    test_recommender(SVD(),
                     "Movielens1m/ml-1m",
                     "Movielens1m",
                     index,
                     max_entries_per_user,
                     "matrix_factorization_predicted_ratings")

print("Movielens 1m done\n\n")


# Loop from 1-5 for the 5 Movielens 10m test sets
for index in range(1, 6):
    test_recommender(KNNBasic(sim_options={'name': 'cosine', 'user_based': True}),
                     "Movielens10m/ml-10M100K",
                     "Movielens10m",
                     index,
                     max_entries_per_user,
                     "knn_predicted_ratings")

    test_recommender(SVD(),
                     "Movielens10m/ml-10M100K",
                     "Movielens10m",
                     index,
                     max_entries_per_user,
                     "matrix_factorization_predicted_ratings")

print("Movielens 10m done")
