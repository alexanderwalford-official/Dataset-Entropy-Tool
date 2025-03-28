
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import statistics
from datetime import datetime
import os
from entropy_methods import *
import math

#! IMPORTANT: Create a config.txt file containing your API keys with the same variable names.

#! TODO: Implement some kind of correlation detection and attempt to adjust the standard deviation range dynamically for each row.

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


# define if you would only like to use up to a certain value
ONLY_USE_ENABLED = True
ONLY_USE = 100
standard_deviation_range = 10 # WARNING: first x values will be removed

# will ignore the standard deviation range value, dynamic fitting
AUTO_CORRELATION = True
CORRELATION_MULTIPLIER = 2
CORRELATION_SCALE = 100
SHOW_CORRELATION_GRAPH = False
DEVIATION_METHOD = "bayesian" # options: std, mad, iqr, bayesian

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


def correlation_sign(lst):
    lst = np.array(lst, dtype=np.float64)
    lst = (lst - np.mean(lst)) / np.std(lst)  # standardise to mean=0, std=1
    corr = np.corrcoef(lst, np.arange(len(lst)))[0, 1]
    if np.isnan(corr):
        return 0  # handle NaN case
    return corr * CORRELATION_SCALE  # scale

def auto_correlation(vals):
    """
    Create a scaled automatic correlation array for the dataset for profiling the trend required.
    """
    print("[ ! ] Using automatic correlation profiling..")
    lc_cor = 0
    auto_correlation_array = []

    for i in range(len(vals)):  # use index-based iteration
        if lc_cor == standard_deviation_range - 1:
            # ensure there are enough values to take the last x
            start_idx = max(0, i - (standard_deviation_range - 1))  # avoid negative index
            last_x_vals = vals[start_idx : i + 1]  # extract last x elements

            if len(last_x_vals) < 2:  # correlation needs at least 2 points
                print("[ ! ] Warning: Not enough values for correlation. Skipping.")
                continue

            # compute correlation
            correlation_profile = correlation_sign(last_x_vals)  
            
            # check for NaN
            if math.isnan(correlation_profile):
                print("[ ! ] Warning: correlation_profile is NaN. Skipping this entry.")
            else:
                auto_correlation_array.append(int(correlation_profile * CORRELATION_MULTIPLIER))
            
            lc_cor = 0  # reset counter
        else:
            lc_cor += 1

    print("[ ! ] Automatic correlation array:")
    print(auto_correlation_array)

    return auto_correlation_array

def api_random_method(vals, method):
    global standard_deviation_range
    new_vals = []
    auto_correlation_array = []
    lc = 0

    if AUTO_CORRELATION:
        auto_correlation_array = auto_correlation(vals)
        # render a line chart
        plt.figure(figsize=(10, 5)) # set the figure size
        plt.plot(auto_correlation_array, label="Value")
        plt.title("Dataset Correlation Profile")
        plt.xlabel('Iteration (sequential time)')
        plt.ylabel('Correlation Amount')
        plt.legend()
        plt.grid(True)
        plt.savefig('output/correlation.png')
        if SHOW_CORRELATION_GRAPH:
            plt.show()
        else:
            plt.close()

    for i in tqdm(vals, total=len(vals), desc="Generating new values using " + method + " noise"):
        if lc > standard_deviation_range - 1: # problem is that first 10 values are not being generated as 10 values are required for standard deviation
            # get next x values
            standard_deviation_range_values_p = []
            lc_n = 0
            for k in vals:
                if lc_n > lc and lc_n < lc + standard_deviation_range:
                    if AUTO_CORRELATION:
                        standard_deviation_range_values_p.append(float(k + auto_correlation_array[int(lc_n / 10)]))
                    else:
                        standard_deviation_range_values_p.append(float(k))
                lc_n += 1

            # compute standard deviation (only if we have enough values)
            if len(standard_deviation_range_values_p) > 1:
                if DEVIATION_METHOD == "std":
                    upper_range_value = statistics.stdev([float(x) for x in standard_deviation_range_values_p])
                elif DEVIATION_METHOD == "mad":
                    upper_range_value = mad_based_std([float(x) for x in standard_deviation_range_values_p])
                elif DEVIATION_METHOD == "iqr":
                    upper_range_value = iqr_based_std([float(x) for x in standard_deviation_range_values_p])
                elif DEVIATION_METHOD == "bayesian":
                    upper_range_value = bayesian_std([float(x) for x in standard_deviation_range_values_p])
                else:
                    ## fallback to std
                    print(["[ X ] Invalid standard deviation method. Using fallback method."])
                    upper_range_value = statistics.stdev([float(x) for x in standard_deviation_range_values_p])
            else:
                upper_range_value = 0

            # get previous 10 values
            standard_deviation_range_values_n = []
            lc_n = 0
            for k in vals:
                if lc_n < lc and lc_n > lc - standard_deviation_range:
                    if AUTO_CORRELATION:
                        standard_deviation_range_values_n.append(float(k + auto_correlation_array[int(lc_n / 10)]))
                    else:
                        standard_deviation_range_values_n.append(float(k))
                lc_n += 1

            # compute standard deviation (only if we have enough values)
            if len(standard_deviation_range_values_n) > 1:
                if DEVIATION_METHOD == "std":
                    lower_range_value = statistics.stdev([float(x) for x in standard_deviation_range_values_n])
                elif DEVIATION_METHOD == "mad":
                    lower_range_value = mad_based_std([float(x) for x in standard_deviation_range_values_n])
                elif DEVIATION_METHOD == "iqr":
                    lower_range_value = iqr_based_std([float(x) for x in standard_deviation_range_values_n])
                elif DEVIATION_METHOD == "bayesian":
                    lower_range_value = bayesian_std([float(x) for x in standard_deviation_range_values_n])
                else:
                    ## fall back to std
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
    print("4) Atmospheric & Quantum (API)") # double pass
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
    elif method == "4":
        method = "Atmospheric"
        new_vals = api_random_method(vals, method)
        method = "Quantum"
        new_vals2 = api_random_method(vals, method)
        method = "Atmospheric Quantum"
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
            if method == "Atmospheric Quantum":
                comparison_array.append(float(old_val) - float(new_vals[lc]) - float(new_vals2[lc]))
            else:
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
    plt.title(COLUMN_NAME.title() + " Dataset Entropy Filtering - " + method + " Noise Method")
    plt.xlabel('Iteration (sequential time)')
    plt.ylabel(COLUMN_NAME)
    plt.grid(True)
    plt.legend()
    plt.savefig('output/new_old_rnd.png')
    plt.show()

    # render a line chart illustrating just the differences
    plt.figure(figsize=(10, 5)) # set the figure size
    plt.plot(comparison_array, label="Difference")
    plt.title(COLUMN_NAME.title() + " Dataset Entropy Filtering (difference) - " + method + " Noise Method")
    plt.axhline(y=np.mean(comparison_array), color='red', linestyle='--', linewidth=2, label="Average")
    plt.axhline(y=0, color='green', linestyle='-', linewidth=2, label="Target")
    plt.xlabel('Iteration (sequential time)')
    plt.ylabel("Random, " + DEVIATION_METHOD.upper() + " RND Gen. Val")
    plt.legend()
    plt.grid(True)
    plt.savefig('output/diff_vals.png')
    plt.show()

    print("[ ! ] Done.")


if __name__ == "__main__":
    main()