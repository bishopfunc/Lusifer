from Recommender import test_recommender
from surprise import KNNBaseline, SVD, NMF, NormalPredictor
from pyspark.sql import functions as F
from Recommender_ALS import test_recommender_spark

# Maximum number of entries from a user that will be used during training, the most recent ones will be used
max_entries_per_user = 40  # If zero or less it will allow an unlimited number to be used in training


def round_results(predictions):
    return [(pred.uid, pred.iid, pred.r_ui, round(pred.est), pred.details) for pred in predictions]


def post_process(predictions):
    return predictions.withColumn('prediction', F.round(F.col('prediction'), 0))


data_path = "D:/Canada/Danial/UoW/Dataset/MovieLens/research_baseline/Movielens1m/ml-1m"


# Loop from 1-5 for the 5 Movielens 100k test sets
for index in range(1, 6):
    test_recommender(KNNBaseline(k=40, sim_options={'name': 'cosine', 'user_based': True}),
                     data_path,
                     "Movielens100K",
                     index,
                     max_entries_per_user,
                     "knn_predicted_ratings",
                     round_results)

    test_recommender(SVD(n_factors=100, n_epochs=20),
                     data_path,
                     "Movielens100K",
                     index,
                     max_entries_per_user,
                     "svd_predicted_ratings",
                     round_results)

    test_recommender(NMF(n_factors=15, n_epochs=50),
                     data_path,
                     "Movielens100K",
                     index,
                     max_entries_per_user,
                     "nmf_predicted_ratings",
                     round_results)

    test_recommender(NormalPredictor(),
                     data_path,
                     "Movielens100K",
                     index,
                     max_entries_per_user,
                     "normal_predicted_ratings",
                     round_results)

    test_recommender_spark(
        data_path,
        "Movielens100K",
        index,
        max_entries_per_user,
        "als_predicted_ratings",
        post_process)

print("Movielens 100k done\n\n")


# # Loop from 1-5 for the 5 Movielens 1m test sets
# for index in range(1, 6):
#     test_recommender(KNNBaseline(k=40, sim_options={'name': 'cosine', 'user_based': True}),
#                      "Movielens1m/ml-1m",
#                      "Movielens1m",
#                      index,
#                      max_entries_per_user,
#                      "knn_predicted_ratings",
#                      round_results)
#
#     test_recommender(SVD(n_factors=100, n_epochs=20),
#                      "Movielens1m/ml-1m",
#                      "Movielens1m",
#                      index,
#                      max_entries_per_user,
#                      "svd_predicted_ratings",
#                      round_results)
#
#     test_recommender(NMF(n_factors=15, n_epochs=50),
#                      "Movielens1m/ml-1m",
#                      "Movielens1m",
#                      index,
#                      max_entries_per_user,
#                      "nmf_predicted_ratings",
#                      round_results)
#
#     test_recommender_spark(
#         "Movielens1m/ml-1m",
#         "Movielens1m",
#         index,
#         max_entries_per_user,
#         "als_predicted_ratings",
#         post_process)
#
# print("Movielens 1m done\n\n")
#
#
# # Loop from 1-5 for the 5 Movielens 10m test sets
# for index in range(1, 6):
#     test_recommender(KNNBaseline(k=40, sim_options={'name': 'cosine', 'user_based': True}),
#                      "Movielens10m/ml-10M100K",
#                      "Movielens10m",
#                      index,
#                      max_entries_per_user,
#                      "knn_predicted_ratings",
#                      round_results)
#
#     test_recommender(SVD(n_factors=100, n_epochs=20),
#                      "Movielens10m/ml-10M100K",
#                      "Movielens10m",
#                      index,
#                      max_entries_per_user,
#                      "svd_predicted_ratings",
#                      round_results)
#
#     test_recommender(NMF(n_factors=15, n_epochs=50),
#                      "Movielens10m/ml-10M100K",
#                      "Movielens10m",
#                      index,
#                      max_entries_per_user,
#                      "nmf_predicted_ratings",
#                      round_results)
#
#     test_recommender_spark(
#         "Movielens10m/ml-10M100K",
#         "Movielens10m",
#         index,
#         max_entries_per_user,
#         "als_predicted_ratings",
#         post_process)
#
# print("Movielens 10m done")
