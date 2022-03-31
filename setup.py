import wget
import pandas as pd
import os.path

SCRIPT_DIR = os.path.dirname(__file__)

DATASET_PATH = os.path.join(SCRIPT_DIR, 'data', 'communities_full.csv')
PROCESSED_DATASET_PATH = os.path.join(SCRIPT_DIR, 'data', 'communities_processed.csv')
DATASET_DESCRIPTION_PATH = os.path.join(SCRIPT_DIR, 'data', 'dataset_description.txt')

# Download data set
# Source: UCI Machine Learning Repository, , Michael Redmond
# http://archive.ics.uci.edu/ml/datasets/communities+and+crime+unnormalized

print("===========  Downloading Dataset ===========\n")

if os.path.exists(DATASET_PATH):
    print("Dataset already download. Skipping download.")
else:
    dataset_url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/00211/CommViolPredUnnormalizedData.txt'
    dataset_description_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/communities/communities.names'

    wget.download(dataset_url, DATASET_PATH)
    wget.download(dataset_description_url, DATASET_DESCRIPTION_PATH)

print("\n")

print("===========  Preprocessing Dataset ===========\n")

selected_column_names = ['population', 'householdsize', 'agePct12t29', 'agePct65up',
                       'medFamInc', 'PctPopUnderPov', 'MedRentPctHousInc', 'TotalPctDiv',
                       'PctLargHouseFam', 'pctUrban', 'PctHousOccup','ViolentCrimesPerPop']

# 'PolicPerPop', 'PoliceBudgPerPop',  108, 126, => don't have complete data
# Only using rows with complete data would bias for bigger communities:
# => 'A limitation was that the LEMAS survey was of the police departments with at least 100 officers, 
# plus a random sample of smaller departments. For our purposes, communities not found in both census and crime datasets 
# were omitted. Many communities are missing LEMAS data.'
# I am not sure if we should strike that bargain and therefore opted for leaving these features out for now.

selected_columns = [5, 6, 12, 14, 24, 33, 91, 46, 67, 16, 77, 127]
print(f"Selecting the following columns: {selected_column_names}\n")

df = pd.read_csv(DATASET_PATH, header=None)
df = df[selected_columns]
df.columns = selected_column_names

print(f"Writing preprocessed dataset to {PROCESSED_DATASET_PATH}\n")
df.to_csv(PROCESSED_DATASET_PATH, index=False)

