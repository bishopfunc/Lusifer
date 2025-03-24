#!/usr/bin/env python
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr


class Evaluator:
    """
    Evaluator class to compute RMSE, MAE, and R2 for each recommender prediction method.
    It also generates publication-quality visualizations and outputs results to CSV.
    """

    def __init__(self, data, methods):
        """
        Initialize the Evaluator by loading the CSV file.

        Parameters:
            csv_file (str): Path to the CSV file containing predictions.
        """
        self.data = data
        # List of method prediction columns in the CSV file
        self.methods = methods
        self.results = None

    def evaluate_metrics(self):
        """
        Compute RMSE, MAE, and R2 for each prediction method.

        Returns:
            pd.DataFrame: DataFrame containing metrics for each method.
        """
        results = []
        ground_truth = self.data['rating']
        for method in self.methods:
            predictions = self.data[method]
            rmse = np.sqrt(mean_squared_error(ground_truth, predictions))
            mae = mean_absolute_error(ground_truth, predictions)
            r2 = r2_score(ground_truth, predictions)
            # Calculate Pearson correlation (only the correlation coefficient)
            corr, _ = pearsonr(ground_truth, predictions)
            results.append({
                "Method": method,
                "RMSE": rmse,
                "MAE": mae,
                "R2": r2,
                "Pearson": corr
            })
        self.results = pd.DataFrame(results)
        return self.results

    def save_results(self, output_csv='evaluation_results.csv'):
        """
        Save the computed evaluation metrics to a CSV file.

        Parameters:
            output_csv (str): Path for the output CSV file.
        """
        if self.results is None:
            self.evaluate_metrics()
        self.results.to_csv(output_csv, index=False)
        print(f"Results saved to {output_csv}")

    def plot_results(self, output_dir='plots'):
        """
        Generate and save bar plots for RMSE, MAE, and R2 metrics.
        The plots are designed to be of high quality for scientific publication.

        Parameters:
            output_dir (str): Directory where the plots will be saved.
        """
        if self.results is None:
            self.evaluate_metrics()
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # List of metrics to plot.
        metrics = ['RMSE', 'MAE', 'R2']
        for metric in metrics:
            plt.figure(figsize=(8, 6))
            # Generate a color array for a visually appealing color scheme.
            colors = plt.cm.viridis(np.linspace(0, 1, len(self.results)))
            plt.bar(self.results['Method'], self.results[metric], color=colors)
            plt.xlabel("Methods", fontsize=12)
            plt.ylabel(metric, fontsize=12)
            plt.title(f"{metric} Comparison for Different Methods", fontsize=14)
            plt.xticks(rotation=45, fontsize=10)
            plt.yticks(fontsize=10)
            plt.tight_layout()
            plot_path = os.path.join(output_dir, f"{metric}_comparison.png")
            plt.savefig(plot_path, dpi=300)
            plt.close()
            print(f"Plot saved to {plot_path}")


if __name__ == "__main__":

    data = "1m"

    test_set = pd.read_csv(f"../outputs/rating_test_df_{data}_local.csv")
    train_set = pd.read_csv(f'../Samples/Data/{data}/u1.base', sep='\t', names=['user_id', 'movie_id', 'rating', 'timestamp'],
                            encoding='latin-1')

    # methods = ["SVD++", "NCF", "RNN4REC", "ALS", "4o_mini", "Gemma3_12B", "Gemma3_4B"]
    methods = ["SVD++", "NCF", "RNN4REC", "ALS", "4o_mini", "Gemma3_4B"]

    # ---------------------------------------------------------------------
    # cold start scenario
    # Count the number of interactions per user in the training set
    user_interaction_counts = train_set['user_id'].value_counts()

    # Identify cold start users: those with fewer than 5 interactions in trainset_df
    cold_start_users = user_interaction_counts[user_interaction_counts < 15].index.tolist()

    # Create a new testset_df that only includes rows for these cold start users
    test_set = test_set[test_set['user_id'].isin(cold_start_users)]

    # ---------------------------------------------------------------------
    # ---------------------------------------------------------------------

    evaluator = Evaluator(test_set, methods)
    evaluator.evaluate_metrics()

    evaluator.save_results(f'evaluation_results_{data}.csv')
    evaluator.plot_results()
