# My first Python program to create an Artificial Intelligence (AI) model
# The AI should predict the survival of passengers on the Titanic

import pandas as pd
import os

# Program Path
cwd = os.getcwd()

# Data Path
data_path = cwd + '/data/'

# Init a global variable (df) to hold the DataFrame
global df


# Function to load data from a CSV file
def load_data(data_file):
    global df
    df = pd.read_csv(data_file)


# Main execution
load_data(data_path + 'train.csv')

# Show missing values in the dataset
print("\nMissing values in the dataset:\n", df.isnull().sum())

# Get the median age and fill missing age values
median_age = df['Age'].median()
df['Age'].fillna(median_age, inplace=True)

# Recheck missing values after filling 'Age'
print("\nMissing values in the dataset after filling 'Age':\n", df.isnull().sum())

# 'Embarked' has some missing values, i will fill them with the most frequent port
most_frequent_port = df['Embarked'].mode()[0]
df['Embarked'].fillna(most_frequent_port, inplace=True)

# Recheck missing values after filling 'Embarked'
print("\nMissing values in the dataset after filling 'Embarked':\n", df.isnull().sum())

# 'Cabin' has too much missing data, i will remove it -> irrepairable
df.drop('Cabin', axis=1, inplace=True)

# Idk how to handle 'Name' and 'Ticket' yet, so i will remove them for now
df.drop(['Name', 'Ticket'], axis=1, inplace=True)

# Recheck missing values after dropping 'Cabin', 'Name', and 'Ticket'
print("\nMissing values in the dataset after all adjustments:\n", df.isnull().sum())

if df.isnull().values.any():
    print("\nThere are still missing values in the dataset, check the code.")
    exit()
else:
    print("\nNo missing values left in the dataset. Beginning with converting Sex and Embarked...")

# I will convert 'Sex' into numerical values (male -> 0, female -> 1) | This makes the data understandable for the AI model
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
print("\n'Sex' converted to numerical values.")

# I will convert 'Embarked' into 'One-Hot' Encoding
# This creates a separate column for each port (0 or 1)
df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)
print("'Embarked' converted to One-Hot Encoding.")

# Last dataset check (hopefully no missing values left)
print("\nSo sehen die finalen Daten für das Modell aus:")
print(df.head())
