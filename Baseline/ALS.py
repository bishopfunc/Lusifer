import warnings

import numpy as np
import implicit
from surprise import AlgoBase
from scipy.sparse import coo_matrix


class ImplicitALSRecommender(AlgoBase):
    def __init__(self, factors=10, regularization=0.1, iterations=15):
        # Initialize the ALS model parameters
        super().__init__()
        self.factors = factors
        self.regularization = regularization
        self.iterations = iterations
        self.model = implicit.als.AlternatingLeastSquares(factors=factors, regularization=regularization,
                                                          iterations=iterations)
        self.defaultCount = 0

    def fit(self, trainset):
        # Save the trainset (required for the test method)
        self.trainset = trainset

        # Convert the Surprise trainset to a sparse matrix format suitable for implicit
        user_items = self._trainset_to_sparse(trainset)

        print(f"Matrix shape before transpose (user-item): {user_items.shape}")
        num_nonzero_entries = user_items.count_nonzero()
        print(f"Number of non-zero entries before transpose: {num_nonzero_entries}")

        # Implicit expects the user-item interaction matrix to be in item-user format
        # So we need to transpose it
        item_users = user_items.T.tocsr()
        print(f"Matrix shape after transpose (item-user): {item_users.shape}")

        # Train the ALS model using explicit ratings directly
        self.model.fit(item_users)

        # Store user and item factors for later use
        self.user_factors = self.model.user_factors
        self.item_factors = self.model.item_factors

        print(f"Number of user factors: {len(self.user_factors)}, Number of item factors: {len(self.item_factors)}")

    import warnings

    def estimate(self, u, i):
        # Initialize default values for inner_u and inner_i
        inner_u = None
        inner_i = None

        try:
            inner_u = self.trainset.to_inner_uid(u)
        except ValueError:
            self.defaultCount += 1
            print(f"Default >> {self.defaultCount}")
            return self.trainset.global_mean  # or another default value

        try:
            inner_i = self.trainset.to_inner_iid(i)
        except ValueError:
            self.defaultCount += 1
            print(f"Default >> {self.defaultCount}")
            return self.trainset.global_mean  # or another default value

        # Check if inner_u or inner_i are out of bounds for the latent factors matrix
        if inner_i >= len(self.item_factors):
            self.defaultCount += 1
            print(f"Default >> {self.defaultCount}")
            print(f"Out of bounds: user {inner_u}, item {inner_i}, len(user_factors)={len(self.user_factors)}, len(item_factors)={len(self.item_factors)}.")
            return self.trainset.global_mean

        # Return the dot product of the user and item latent factors
        return np.dot(self.user_factors[inner_u], self.item_factors[inner_i])

    def _trainset_to_sparse(self, trainset):
        """Convert the Surprise Trainset to a sparse matrix for implicit."""
        # Get the number of users and items
        num_users = trainset.n_users
        num_items = trainset.n_items

        # Build lists of user indices, item indices, and ratings
        users = []
        items = []
        ratings = []

        for u, i, r in trainset.all_ratings():
            users.append(u)
            items.append(i)
            ratings.append(r)

        # Create a sparse matrix in COO format
        return coo_matrix((ratings, (users, items)), shape=(num_users, num_items))
