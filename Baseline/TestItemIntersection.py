import pandas as pd


def get_total_movies(file1, file2, separator='\t'):
    column_names = ['userId', 'movieId', 'rating', 'timestamp']

    # Load the training and test datasets
    train_df = pd.read_csv(file1, sep=separator, names=column_names)
    test_df = pd.read_csv(file2, sep=separator, names=column_names)

    # Get the unique userIds from both datasets
    train_movies = set(train_df['movieId'].unique())
    test_movies = set(test_df['movieId'].unique())

    # Find the intersection of userIds
    common_movies = train_movies.union(test_movies)

    return len(common_movies)


def get_movies(file):
    column_names = ['userId', 'movieId', 'rating', 'timestamp']

    # Load the training and test datasets
    df = pd.read_csv(file, sep='\t', names=column_names)

    # Get the unique userIds from both datasets
    movies = set(df['movieId'].unique())

    return len(movies)


def get_intersection(file1, file2, separator='\t'):
    column_names = ['userId', 'movieId', 'rating', 'timestamp']

    # Load the training and test datasets
    train_df = pd.read_csv(file1, sep=separator, names=column_names)
    test_df = pd.read_csv(file2, sep=separator, names=column_names)

    # Get the unique userIds from both datasets
    train_movies = set(train_df['movieId'].unique())
    test_movies = set(test_df['movieId'].unique())

    # Find the intersection of userIds
    common_movies = train_movies.intersection(test_movies)

    return len(common_movies)

    # Output the number of common userIds
    #print(f'Number of common userIds: {len(common_movies)}')


def output_folder_intersections(folder_path, separator='\t'):
    print(f'Path: {folder_path}')
    for i in range(1,6):
        intersection = get_intersection(f'{folder_path}/u{i}.base', f'{folder_path}/u{i}.test', separator=separator)
        # total = get_total_movies(f'{folder_path}/u{i}.base', f'{folder_path}/u{i}.test', separator=separator)
        total = get_movies(f'{folder_path}/u{i}.test')
        print(f'Common movies in file {i}: {intersection} / {total} = {intersection / total * 100}%')
    print('\n')


output_folder_intersections('Movielens100K/ml-100k')
output_folder_intersections('Movielens1m/ml-1m')
output_folder_intersections('Movielens10m/ml-10M100K')
