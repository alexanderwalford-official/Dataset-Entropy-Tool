
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import statistics
from datetime import datetime
import os
from entropy_methods import *

#! IMPORTANT: Create a config.txt file containing your API keys with the same variable names.


def read_config(value):
    """
    Reads the config.txt file for the API keys required to operate the endpoints.
    """
    filename="config.txt"
    with open(filename, "r") as file:
        for line in file:
            if line.startswith(value):
                return line.strip().split("=", 1)[1]
    print("[ X ] Could not find a valid value in configuration file. Are you sure that you created the config.txt file and populated it?")
    return None 

# load config
DATASET_FILE = "2024-02-11__2024-04-08.csv"
COLUMN_NAME = "time"
RANDOM_API_KEY = read_config("RANDOM_API_KEY")
QUANTUM_API_KEY = read_config("QUANTUM_API_KEY")
standard_deviation_range = 10 # WARNING: first x values will be removed

# define if you would only like to use up to a certain value
ONLY_USE_ENABLED = True
ONLY_USE = 100

#! do not modify these values
method = "" # leave as blank

def main():
    print("Dataset Entropy Tool")
    print("Alexander Walford 2025")
    print("\n")
    print("Please select an option from below:\n")
    print("1) Load and generate new dataset.")
    print("2) Exit application.")
    usr_in = input()
    if usr_in == "1":
        load_csv()
    elif usr_in == "2":
        return
    else:
        print("[ X ] Invalid value, try again.")
        input()
        main()

def api_random_method(vals, method):
    new_vals = []
    lc = 0

    for i in tqdm(vals, total=len(vals), desc="Generating new values"):
        if lc > standard_deviation_range - 1: # problem is that first 10 values are not being generated as 10 values are required for standard deviation
            # get next x values
            standard_deviation_range_values_p = []
            lc_n = 0
            for k in vals:
                if lc_n > lc and lc_n < lc + standard_deviation_range:
                    standard_deviation_range_values_p.append(float(k))
                lc_n += 1

            # compute standard deviation (only if we have enough values)
            if len(standard_deviation_range_values_p) > 1:
                upper_range_value = statistics.stdev([float(x) for x in standard_deviation_range_values_p])
            else:
                upper_range_value = 0

            # get previous 10 values
            standard_deviation_range_values_n = []
            lc_n = 0
            for k in vals:
                if lc_n < lc and lc_n > lc - standard_deviation_range:
                    standard_deviation_range_values_n.append(float(k))
                lc_n += 1

            # compute standard deviation (only if we have enough values)
            if len(standard_deviation_range_values_n) > 1:
                lower_range_value = statistics.stdev([float(x) for x in standard_deviation_range_values_n])
            else:
                lower_range_value = 0

            # fetch atmospheric noise and store result
            if method == "Atmospheric":
                new_row = int(fetch_atmospheric_noise(1, lower_range_value, upper_range_value, RANDOM_API_KEY))
                new_vals.append(new_row)
            elif method == "Quantum":
                new_row = int(fetch_quantum_entropy(1, lower_range_value, upper_range_value, QUANTUM_API_KEY))
                new_vals.append(new_row)
        else:
            # add blank value, first x will be removed anyway
            new_vals.append(0)

        lc += 1
    return new_vals


def gaussian_noise_method(vals):
    new_vals = []
    lc = 0

    for i in tqdm(vals, total=len(vals), desc="Generating new values with Gaussian Noise"):
        if lc > standard_deviation_range - 1:
            # get the previous x values for calculating mean and standard deviation
            standard_deviation_range_values_n = [float(k) for k in vals[max(0, lc - standard_deviation_range):lc]]
            
            if len(standard_deviation_range_values_n) > 1:
                mean_value = np.mean(standard_deviation_range_values_n)
                stddev_value = np.std(standard_deviation_range_values_n)
            else:
                mean_value = 0
                stddev_value = 1  # default standard deviation if not enough data
            
            # generate Gaussian noise based on the computed mean and stddev
            new_row = fetch_gaussian_noise(1, mean_value, stddev_value)
            new_vals.append(new_row[0])
        else:
            # add blank value, first x will be removed anyway
            new_vals.append(0)

        lc += 1

    return new_vals

def load_csv():
    dataset_obj = pd.read_csv(DATASET_FILE)
    vals = []
    dt_size = len(dataset_obj)

    print("Dataset size: " + str(dataset_obj.size))

    lc = 0
    for _, row in tqdm(dataset_obj.iterrows(), total=dt_size, desc="Processing rows"):
        if ONLY_USE_ENABLED and lc < ONLY_USE or ONLY_USE_ENABLED == False:
            # check if needs to be specifically converted
            if COLUMN_NAME == "time":
                datetime_obj = datetime.strptime(row[COLUMN_NAME], "%Y-%m-%dT%H:%M:%S.%fZ")
                timestamp = int(datetime_obj.timestamp())
                vals.append(timestamp)
            else:
                vals.append(float(row[COLUMN_NAME])) 
            lc = lc + 1

    print("\n\nPlease select your method of entropy:")
    print("1) Atmospheric Noise (API)")
    print("2) Gaussian Noise (local)")
    print("3) Quantum Noise (API)")
    print("\n> ")

    method = input()
    new_vals = None

    if method == "1":
        # call the relevant method
        method = "Atmospheric"
        new_vals = api_random_method(vals, method)
    elif method == "2":
        method = "Gaussian"
        new_vals = gaussian_noise_method(vals)
    elif method == "3":
        method = "Quantum"
        new_vals = api_random_method(vals, method)
    else:
        print("[ X ] Invalid value.")
        input()
        load_csv()

    # now remove first x values that cannot be compared
    vals = vals[standard_deviation_range:]
    new_vals = new_vals[standard_deviation_range:]

    # print arrays
    print("Original values (" + str(len(new_vals)) + "):")
    print(vals)

    print("New values (" + str(len(new_vals)) + "):")
    print(new_vals)

    # now convert the new_vals into a dataframe
    new_vals_df = pd.DataFrame(new_vals)

    # check if output folder exists, if it doesn't then create it!
    if os.path.exists("output") == False:
        os.makedirs("output")

    # save into new csv
    new_vals_df.to_csv("output/random_values.csv", index=False)

    # compare original vals and new vals and generate a difference array
    comparison_array = []
    lc = 0
    for old_val in vals:
        if lc < len(new_vals):
            comparison_array.append(float(old_val) - float(new_vals[lc]))
        else:
            print("[ X ] Invalid length, new and old do not match.")
            comparison_array.append(float(old_val))
        lc += 1

    print(comparison_array)

    # convert the comparisons array into a dataframe
    comparison_array_df = pd.DataFrame(comparison_array)

    # save into a new csv
    comparison_array_df.to_csv("output/comparisons.csv", index=False)

    # normalize vals
    vals_min = min(vals)
    vals_max = max(vals)
    normalized_vals = [(x - vals_min) / (vals_max - vals_min) for x in vals]

    # normalize new_vals
    new_vals_min = min(new_vals)
    new_vals_max = max(new_vals)
    normalized_new_vals = [(x - new_vals_min) / (new_vals_max - new_vals_min) for x in new_vals]

    # render a line chart containing the original and new values using matplotlib
    plt.plot(normalized_vals, label="Original Values")
    plt.plot(normalized_new_vals, label="New Random Values")
    plt.title(COLUMN_NAME + "Dataset Entropy Filtering - " + method + " Noise Method")
    plt.xlabel('Iteration (sequential time)')
    plt.ylabel(COLUMN_NAME)
    plt.grid(True)
    plt.legend()
    plt.savefig('output/new_old_rnd.png')
    plt.show()

    # render a line chart illustrating just the differences
    plt.figure(figsize=(10, 5)) # set the figure size
    plt.plot(comparison_array, label="Difference")
    plt.title(COLUMN_NAME + "Dataset Entropy Filtering (difference) - " + method + " Noise Method")
    plt.xlabel('Iteration (sequential time)')
    plt.ylabel('Random, STD RND Gen. Val')
    plt.legend()
    plt.grid(True)
    plt.savefig('output/diff_vals.png')
    plt.show()

    print("[ ! ] Done.")


if __name__ == "__main__":
    main()