import time

import pandas as pd
import tensorflow as tf

from libreco.algorithms import ALS, NCF, RNN4Rec, SVDpp
from libreco.data import DatasetPure, split_by_ratio_chrono, split_by_ratio


def reset_state(name):
    tf.compat.v1.reset_default_graph()
    print("\n", "=" * 30, name, "=" * 30)

def get_ratings(model, model_name, testset_df):
    """
    For each user, call prediction for all items in the test set.
    Build and return a DataFrame with columns:
        user_id | movie_id | recommendation_rank | module_source
    """
    col = f"{model_name}_rate"
    testset_df[col] = None

    for idx, row in testset_df.iterrows():
        rate = model.predict(user=row["user"], item=row["item"])
        testset_df.at[idx, col] = round(rate[0])

    return testset_df


if __name__ == "__main__":
    data = "1m"
    Path = f"../../Samples/Data/{data}"
    epochs = 30

    # Load the ratings data
    trainset_df = pd.read_csv(f'{Path}/u1.base',
                              sep='\t',
                              names=['user_id', 'movie_id', 'rating', 'timestamp'],
                              encoding='latin-1')

    # Sort by userId and timestamp to ensure recent entries are at the end
    trainset_df.sort_values(by=['user_id', 'timestamp'], inplace=True)
    # Group by userId and select the n most recent entries for each user
    trainset_df = trainset_df.groupby('user_id').tail(40)

    testset_df = pd.read_csv(f'{Path}/u1.test',
                             sep='\t',
                             names=['user_id', 'movie_id', 'rating', 'timestamp'],
                             encoding='latin-1')

    # ------------------------------------------------
    # ------------------------------------------------
    # 1) Rename columns
    trainset_df.rename(
        columns={
            "user_id": "user",
            "movie_id": "item",
            "rating": "label",
            "timestamp": "time"
        },
        inplace=True
    )

    testset_df.rename(
        columns={
            "user_id": "user",
            "movie_id": "item",
            "rating": "label",
            "timestamp": "time"
        },
        inplace=True
    )

    # 2) Re-order columns if needed (e.g., drop the timestamp or put it last)
    trainset_df = trainset_df[["user", "item", "label", "time"]]
    testset_df = testset_df[["user", "item", "label", "time"]]

    # Suppose after renaming columns to 'user' and 'item'
    train_users = set(trainset_df["user"].unique())
    train_items = set(trainset_df["item"].unique())

    # Filter test so it only has user/item that appear in train
    testset_df = testset_df[
        testset_df["user"].isin(train_users) & testset_df["item"].isin(train_items)
        ]

    # Ensure users and items in the test set exist in the training set
    train_users = set(trainset_df["user"].unique())
    train_items = set(trainset_df["item"].unique())

    # Filter test set to include only known users and items
    testset_df = testset_df[
        testset_df["user"].isin(train_users) & testset_df["item"].isin(train_items)
        ]

    # Create a set of user-item pairs from the training set
    train_user_item_pairs = set(zip(trainset_df["user"], trainset_df["item"]))

    # Remove any user-item interactions from the test set that exist in the training set
    testset_df = testset_df[
        ~testset_df.apply(lambda row: (row["user"], row["item"]) in train_user_item_pairs, axis=1)
    ]

    # ------------------------------------------------
    # ------------------------------------------------

    # Build train and eval sets
    # train_data, eval_data = split_by_ratio(trainset_df, test_size=0.1)
    train_data, data_info = DatasetPure.build_trainset(trainset_df)
    # eval_data = DatasetPure.build_evalset(eval_data)
    print(data_info)

    # =========================== load model ==============================
    # ----------------------------------------------------------------

    start_time = time.perf_counter()
    metrics = ["rmse", "mae", "r2"]


    reset_state("SVD++")
    svdpp = SVDpp(
        task="rating",
        data_info=data_info,
        embed_size=16,
        n_epochs=epochs,
        lr=0.001,
        reg=None,
        batch_size=256,
    )
    svdpp.fit(
        train_data,
        neg_sampling=False,
        verbose=2,
        shuffle=True,
        # eval_data=eval_data,
        metrics=metrics,
    )

    testset_df = get_ratings(model=svdpp, model_name="SVD++", testset_df=testset_df)

    # ----------------------------------------------------------------
    # ----------------------------------------------------------------

    reset_state("NCF")
    ncf = NCF(
        "rating",
        data_info,
        embed_size=16,
        n_epochs=epochs,
        lr=0.001,
        lr_decay=False,
        reg=None,
        batch_size=256,
        num_neg=1,
        use_bn=True,
        dropout_rate=None,
        hidden_units=(128, 64, 32),
        tf_sess_config=None,
    )
    ncf.fit(
        train_data,
        neg_sampling=False,
        verbose=2,
        shuffle=True,
        # eval_data=eval_data,
        metrics=metrics,
    )
    testset_df = get_ratings(model=ncf, model_name="NCF", testset_df=testset_df)

    #---------------------------------------------------------------
    #---------------------------------------------------------------

    reset_state("RNN4Rec")
    rnn = RNN4Rec(
        "rating",
        data_info,
        rnn_type="lstm",
        embed_size=16,
        n_epochs=epochs,
        lr=0.001,
        lr_decay=False,
        hidden_units=(16, 16),
        reg=None,
        batch_size=256,
        num_neg=1,
        dropout_rate=None,
        recent_num=10,
        tf_sess_config=None,
    )
    rnn.fit(
        train_data,
        neg_sampling=False,
        verbose=2,
        shuffle=True,
        # eval_data=eval_data,
        metrics=metrics,
    )

    testset_df = get_ratings(model=rnn, model_name="RNN4REC", testset_df=testset_df)

    reset_state("ALS")
    als = ALS(
        task="rating",
        data_info=data_info,
        embed_size=16,
        n_epochs=epochs,
        reg=5.0,
        alpha=10,
        use_cg=False,
        n_threads=1,
        seed=42,
    )
    als.fit(
        train_data,
        neg_sampling=False,
        verbose=2,
        shuffle=True,
        # eval_data=eval_data,
        metrics=metrics,
    )

    testset_df = get_ratings(model=als, model_name="ALS", testset_df=testset_df)
    testset_df.to_csv("testset_df.csv")

