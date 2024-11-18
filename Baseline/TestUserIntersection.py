import pandas as pd


def get_total_users(file1, file2, separator='\t'):
    column_names = ['userId', 'movieId', 'rating', 'timestamp']

    # Load the training and test datasets
    train_df = pd.read_csv(file1, sep=separator, names=column_names)
    test_df = pd.read_csv(file2, sep=separator, names=column_names)

    # Get the unique userIds from both datasets
    train_users = set(train_df['userId'].unique())
    test_users = set(test_df['userId'].unique())

    # Find the intersection of userIds
    common_users = train_users.union(test_users)

    return len(common_users)


def get_users(file):
    column_names = ['userId', 'movieId', 'rating', 'timestamp']

    # Load the training and test datasets
    df = pd.read_csv(file, sep='\t', names=column_names)

    # Get the unique userIds from both datasets
    users = set(df['userId'].unique())

    return len(users)


def get_intersection(file1, file2, separator='\t'):
    column_names = ['userId', 'movieId', 'rating', 'timestamp']

    # Load the training and test datasets
    train_df = pd.read_csv(file1, sep=separator, names=column_names)
    test_df = pd.read_csv(file2, sep=separator, names=column_names)

    # Get the unique userIds from both datasets
    train_users = set(train_df['userId'].unique())
    test_users = set(test_df['userId'].unique())

    # Find the intersection of userIds
    common_users = train_users.intersection(test_users)

    return len(common_users)

    # Output the number of common userIds
    #print(f'Number of common userIds: {len(common_users)}')


def output_folder_intersections(folder_path, separator='\t'):
    print(f'Path: {folder_path}')
    for i in range(1,6):
        intersection = get_intersection(f'{folder_path}/u{i}.base', f'{folder_path}/u{i}.test', separator=separator)
        # total = get_total_users(f'{folder_path}/u{i}.base', f'{folder_path}/u{i}.test', separator=separator)
        total = get_users(f'{folder_path}/u{i}.test')
        print(f'Common users in file {i}: {intersection} / {total} = {intersection / total * 100}%')
    print('\n')


output_folder_intersections('Movielens100K/ml-100k')
output_folder_intersections('Movielens1m/ml-1m')
output_folder_intersections('Movielens10m/ml-10M100K')
