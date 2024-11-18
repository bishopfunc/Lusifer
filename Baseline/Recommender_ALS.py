import os
import warnings

import pandas as pd
from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS  # Import only the required ALS class
from pyspark.sql.window import Window
from pyspark.sql import functions as F
from pyspark.ml.evaluation import RegressionEvaluator


def test_recommender_spark(data_path, output_path, index, n_recent_entries, heading_name, post_process):
    print(f"Running test with index {index} to produce {heading_name}")

    # Paths for base and test files
    train_file = data_path + f'/u{index}.base'
    test_file = data_path + f'/u{index}.test'

    # Start Spark session
    spark = SparkSession.builder.appName("ALSRecommenderTest").getOrCreate()

    # Load the train and test data using Spark DataFrames
    train_df = spark.read.csv(train_file, sep='\t', inferSchema=True, header=False)\
                         .toDF('userId', 'movieId', 'rating', 'timestamp')
    test_df = spark.read.csv(test_file, sep='\t', inferSchema=True, header=False)\
                        .toDF('userId', 'movieId', 'rating', 'timestamp')

    # Reduce the data to the n most recent if it is a positive number, otherwise use the whole dataset
    if n_recent_entries > 0:
        # Sort by userId and timestamp to ensure recent entries are at the end
        window = Window.partitionBy('userId').orderBy(F.desc('timestamp'))
        train_df = train_df.withColumn('rank', F.row_number().over(window))\
                           .filter(F.col('rank') <= n_recent_entries).drop('rank')

    print("Training...")
    # Set up ALS model
    als = ALS(userCol='userId', itemCol='movieId', ratingCol='rating', rank=10, maxIter=10, regParam=0.01, coldStartStrategy="nan")

    # Train the ALS model
    als_model = als.fit(train_df)

    print("Testing...")
    # Make predictions on the test set
    predictions = als_model.transform(test_df)

    if predictions.filter(F.col('prediction').isNull() | F.isnan(F.col('prediction'))).count() > 0:
        warnings.warn("Warning: Some predictions were set to the average rating due to cold start issues. count=" + str(predictions.filter(F.col('prediction').isNull() | F.isnan(F.col('prediction'))).count()))
        average_rating = round(train_df.agg(F.avg('rating')).first()[0])
        predictions = predictions.withColumn(
            'prediction',
            F.when(F.isnan(F.col('prediction')) | F.col('prediction').isNull(), average_rating)
            .otherwise(F.col('prediction'))
        )

        print("Count after using average. count=" + str(predictions.filter(F.col('prediction').isNull() | F.isnan(F.col('prediction'))).count()))

    # Evaluate the model
    evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")
    rmse = evaluator.evaluate(predictions)
    print(f"RMSE: {rmse}")

    # Post-process predictions
    processed_predictions = post_process(predictions)

    # Convert predictions to Pandas for final merging
    predictions_pd = processed_predictions.select('userId', 'movieId', 'prediction').toPandas()

    # Prepare a DataFrame for the predictions
    predictions_df = predictions_pd.rename(columns={
                        'userId': 'user_id',
                        'movieId': 'movie_id',
                        'prediction': heading_name
                    })

    # Load existing csv file
    existing_csv_file_path = output_path + f'/rating_test_df_u{index}.csv'

    # Check if the file exists
    if os.path.exists(existing_csv_file_path):
        # Load existing csv file
        output_df = pd.read_csv(existing_csv_file_path)

        # Remove previous test data if it exists
        if heading_name in output_df.columns:
            output_df = output_df.drop(columns=[heading_name])
    else:
        # Create a new DataFrame with the same columns as predictions_df
        output_df = test_df.toPandas().rename(columns={
                        'userId': 'user_id',
                        'movieId': 'movie_id'
                    })
        output_df.insert(0, '', range(1, len(output_df) + 1))

    # Merge the predictions with the existing output DataFrame using 'user_id' and 'movie_id'
    merged_df = pd.merge(output_df, predictions_df, on=['user_id', 'movie_id'], how='left')

    # Save the updated DataFrame back to the CSV file
    merged_df.to_csv(existing_csv_file_path, index=False)

    print(f"Updated CSV file saved to {existing_csv_file_path}\n")

    # Stop the Spark session
    spark.stop()
