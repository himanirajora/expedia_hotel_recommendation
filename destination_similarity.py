import os
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

INPUT_DIR = '/home/ec2-user/assignment/expedia/input'
MODELS_DIR = '/home/ec2-user/assignment/expedia/models'
OUTPUT_DIR = '/home/ec2-user/assignment/expedia/output'

TRAIN_FILE = os.path.join(INPUT_DIR, 'train_final.csv')
TEST_FILE = os.path.join(INPUT_DIR, 'test_clean.csv')
DESTINATIONS_FILE = os.path.join(INPUT_DIR, 'destinations.csv')

NROWS = 100


def main():
    df = pd.read_csv(TRAIN_FILE, nrows=NROWS)
    df = df[df['is_booking'] == 1][
        ['user_id', 'srch_destination_id', 'is_booking']]

    # only users who have more than 1 booking
    grouped_users_df = df.groupby(
        ['user_id']).srch_destination_id.nunique().reset_index()
    u = grouped_users_df[
        grouped_users_df['srch_destination_id'] > 1].user_id.tolist()
    del grouped_users_df

    df = df[df['user_id'].isin(u)]
    del u

    grouped_df = df.groupby(
        ['user_id', 'srch_destination_id']).sum().reset_index()

    pivoted_df = grouped_df.pivot(index='srch_destination_id',
                                  columns='user_id', values='is_booking')
    del grouped_df

    pivoted_df.fillna(0, inplace=True)
    destination_similarity = cosine_similarity(pivoted_df)
    del pivoted_df

    destination_similarity_df = pd.DataFrame(destination_similarity)
    destination_similarity_df.to_csv(
        os.path.join(OUTPUT_DIR, 'destination_similarity.csv'))

    # Set of dest ids with missing latent features
    destination_latent_df = pd.read_csv(DESTINATIONS_FILE, nrows=NROWS)
    dest_ids_latent_set = set(destination_latent_df.reset_index()[
                                  'srch_destination_id'])

    dest_ids_main_set = set(df['srch_destination_id'])
    del df

    dest_ids_left = dest_ids_main_set.difference(dest_ids_latent_set)
    print('Total destinations to impute: {}'.format(len(dest_ids_left)))

    # Generate features for dest ids with no features
    latent_df_mean = destination_latent_df.mean()
    new_df = pd.DataFrame(columns=destination_latent_df.columns)
    imputed_by_mean = []
    imputed_by_similar = []
    for dest_id in dest_ids_left:
        try:
            ten = np.argsort(destination_similarity_df.loc[dest_id, :])[
                  -10:].tolist()
            temp = []
            for i in ten:
                if destination_similarity_df.loc[dest_id, str(i)] > 0:
                    temp.append(i)
            if temp:
                mean = new_df.append(
                    destination_latent_df[destination_latent_df[
                        'srch_destination_id'].isin(temp)].mean())
                imputed_by_similar.append(dest_id)
            else:
                mean = latent_df_mean
                imputed_by_mean.append(dest_id)

        except KeyError:
            mean = latent_df_mean
            imputed_by_mean.append(dest_id)

        mean['srch_destination_id'] = dest_id

        destination_latent_df.append([mean], ignore_index=True)

    destination_latent_df.to_csv(
        os.path.join(OUTPUT_DIR, 'destinations_imputed.csv'), index=False)
    pd.DataFrame(imputed_by_similar).to_csv(
        os.path.join(OUTPUT_DIR, 'dest_imputed_by_similar.csv'), index=False)
    pd.DataFrame(imputed_by_mean).to_csv(
        os.path.join(OUTPUT_DIR, 'dest_imputed_by_overall.csv'), index=False)
    print("Total imputed by overall mean: {}".format(len(imputed_by_mean)))
    print("Total imputed by similar mean: {}".format(len(imputed_by_similar)))


if __name__ == '__main__':
    main()
