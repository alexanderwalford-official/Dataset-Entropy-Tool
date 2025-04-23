import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime


NEW_DATASET = "output/comparisons.csv" # replace with the actual path to your new dataset
ORIGINAL_DATASET = "2024-02-11__2024-04-08.csv" # replace with the actual path to your original dataset


def compare_datasets(NEW_DATASET_COLUMN_NAME="0", ORIGINAL_DATASET_COLUMN_NAME="time"):
    """
        Compares two datasets and presents the differences.
    """

    # load the datasets
    original_dataset = pd.read_csv(ORIGINAL_DATASET)
    new_dataset = pd.read_csv(NEW_DATASET)

    # extract the relevant columns
    original_column = original_dataset[ORIGINAL_DATASET_COLUMN_NAME]
    new_column = new_dataset[NEW_DATASET_COLUMN_NAME]

    formatted_original_col = []

    # convert the columns to datetime format
    for row in original_column:
        if ORIGINAL_DATASET_COLUMN_NAME == "time":
            datetime_obj = datetime.strptime(row, "%Y-%m-%dT%H:%M:%S.%fZ")
            timestamp = int(datetime_obj.timestamp())
            formatted_original_col.append(timestamp)
        else:
            formatted_original_col.append(row)

    # remove any extra values from the original column to match the length of the new column and to align the data
    formatted_original_col = formatted_original_col[:len(new_column)]

    print(formatted_original_col)

    # format the new column to match the original column
    formatted_new_col = []
    for row in new_column:
        if NEW_DATASET_COLUMN_NAME == "time":
            datetime_obj = datetime.strptime(row, "%Y-%m-%dT%H:%M:%S.%fZ")
            timestamp = int(datetime_obj.timestamp())
            formatted_new_col.append(timestamp)
        else:
            formatted_new_col.append(row)

    print(formatted_new_col)

    # normalize the datasets
    original_min, original_max = min(formatted_original_col), max(formatted_original_col)
    new_min, new_max = min(formatted_new_col), max(formatted_new_col)

    normalized_original_col = [(x - original_min) / (original_max - original_min) for x in formatted_original_col]
    normalized_new_col = [(x - new_min) / (new_max - new_min) for x in formatted_new_col]

    # plot normalized datasets
    plt.figure(figsize=(10, 5))
    plt.plot(normalized_original_col, label='Original Dataset', color='blue')
    plt.plot(normalized_new_col, label='New Dataset', color='red')
    plt.title('Comparison of Datasets (NORMALIZED)')
    plt.xlabel('Index')
    plt.ylabel('Values')
    plt.legend()
    plt.grid()
    plt.savefig('output/NORMALIZED_FINAL_dataset_comparison.png')
    plt.show()

    # plot non-normalized datasets
    plt.figure(figsize=(10, 5))
    plt.plot(formatted_original_col, label='Original Dataset', color='blue')
    plt.plot(formatted_new_col, label='New Dataset', color='red')
    plt.title('Comparison of Datasets')
    plt.xlabel('Index')
    plt.ylabel('Values')
    plt.legend()
    plt.grid()
    plt.savefig('output/FINAL_dataset_comparison.png')
    plt.show()

    # scatter plot of NORMALIZED original vs new values
    plt.figure(figsize=(10, 5))
    plt.scatter(normalized_original_col, normalized_new_col, color='orange', label='Data Points')
    plt.plot([min(normalized_original_col), max(normalized_original_col)],
            [min(normalized_original_col), max(normalized_original_col)],
            color='blue', linestyle='--', label='Perfect Match (y=x)')
    plt.title('Scatter Plot of Original vs New Dataset')
    plt.xlabel('Original Dataset (NORMALIZED)')
    plt.ylabel('New Dataset')
    plt.legend()
    plt.grid()
    plt.savefig('output/NORMALIZED_dataset_diff_scatter.png')
    plt.show()

    # scatter plot of original vs new values
    plt.figure(figsize=(10, 5))
    plt.scatter(formatted_original_col, formatted_new_col, color='orange', label='Data Points')
    plt.plot([min(formatted_original_col), max(formatted_original_col)],
            [min(formatted_original_col), max(formatted_original_col)],
            color='blue', linestyle='--', label='Perfect Match (y=x)')
    plt.title('Scatter Plot of Original vs New Dataset')
    plt.xlabel('Original Dataset')
    plt.ylabel('New Dataset')
    plt.legend()
    plt.grid()
    plt.savefig('output/dataset_diff_scatter.png')
    plt.show()

    print("[ ! ] Done comparing datasets.")

