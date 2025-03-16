
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import requests
import statistics

# IMPORTANT: Create a config.txt file containing your API keys with the same variable names.


def read_config(value):
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
QRNG_API_KEY = ""
standard_deviation_range = 10


def main():
    print("Dataset Entropy Tool")
    print("Alexander Walford 2025")
    print("\n\n")
    print("Please select an option from below:\n")
    print("1) Load and generate new dataset")
    usr_in = input()
    if usr_in == "1":
        load_csv()
    return

def atmospheric_random_method(vals):
    new_vals = []
    lc = 0

    for i in tqdm(vals, total=len(vals), desc="Generating new values"):
        if lc > 9:  # skip first 10 values
            # get next 10 values
            standard_deviation_range_values_p = []
            lc_n = 0
            for k in vals:
                if lc_n > lc and lc_n < lc + 10:
                    standard_deviation_range_values_p.append(k)
                lc_n += 1

            # compute standard deviation (only if we have enough values)
            if len(standard_deviation_range_values_p) > 1:
                upper_range_value = statistics.stdev(standard_deviation_range_values_p)
            else:
                upper_range_value = 0  # Default if not enough values

            # get previous 10 values
            standard_deviation_range_values_n = []
            lc_n = 0
            for k in vals:
                if lc_n < lc and lc_n > lc - 10:
                    standard_deviation_range_values_n.append(k)
                lc_n += 1

            # compute standard deviation (only if we have enough values)
            if len(standard_deviation_range_values_n) > 1:
                lower_range_value = statistics.stdev(standard_deviation_range_values_n)
            else:
                lower_range_value = 0  # Default if not enough values

            # fetch atmospheric noise and store result
            new_row = fetch_atmospheric_noise(1, lower_range_value, upper_range_value)
            new_vals.append(new_row)

        lc += 1

    return new_vals

def load_csv():
    dataset_obj = pd.read_csv(DATASET_FILE)
    vals = []
    dt_size = len(dataset_obj)

    print("Dataset size: " + str(dataset_obj.size))

    for _, row in tqdm(dataset_obj.iterrows(), total=dt_size, desc="Processing rows"):
        vals.append(row[COLUMN_NAME])

    print("\n\nPlease select your method of entropy:")
    print("1) Atmospheric Noise (random.org)")

    method = input()
    new_vals = None

    if method == "1":
        # call the relevant method
        new_vals = atmospheric_random_method(vals)
    else:
        print("[ X ] Invalid value.")
        input()
        load_csv()

    # print arrays
    print("Original dataset:")
    print(vals)

    print("New values:")
    print(new_vals)

    # now convert the new_vals into a dataframe
    new_vals_df = pd.DataFrame(new_vals)

    # save into new csv
    new_vals_df.to_csv("output/random_values.csv", index=False)

    # compare original vals and new vals and generate a difference array
    comparison_array = []
    lc = 0
    for old_val in vals:
        comparison_array.append(float(old_val) - float(new_vals[lc]))
        lc = lc + 1

    # convert the comparisons array into a dataframe
    comparison_array_df = pd.DataFrame(comparison_array)

    # save into a new csv
    comparison_array_df.to_csv("output/comparisons.csv", index=False)

    # render a line chart containing the original and new values using matplotlib
    plt.figure(figsize=(10, 5)) # set the figure size
    plt.plot(vals, label="Original Values")
    plt.plot(new_vals, label="New Random Values")
    plt.title('Dataset Entropy Filtering - Atmospheric Noise Method')
    plt.xlabel('Iteration (sequential time)')
    plt.ylabel(COLUMN_NAME)
    plt.grid(True)
    plt.savefig('output/new_old_rnd.png')
    plt.show()

    # render a line chart illustrating just the differences
    plt.figure(figsize=(10, 5)) # set the figure size
    plt.plot(comparison_array, label="DIfference")
    plt.title('Dataset Entropy Filtering (difference) - Atmospheric Noise Method')
    plt.xlabel('Iteration (sequential time)')
    plt.ylabel('Random, STD RND Gen. Val')
    plt.grid(True)
    plt.savefig('output/diff_vals.png')
    plt.show()

    print("[ ! ] Done.")


def fetch_atmospheric_noise(num_values, min_val, max_val):
    url = "https://api.random.org/json-rpc/4/invoke"
    headers = {'Content-Type': 'application/json'}
    payload = {
        "jsonrpc": "2.0",
        "method": "generateIntegers",
        "params": {
            "apiKey": RANDOM_API_KEY,
            "n": num_values,
            "min": min_val,
            "max": max_val,
            "replacement": True
        },
        "id": 42
    }
    # try to make the request
    try:
        response = requests.post(url, json=payload, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()
        # check for errors
        if "error" in data:
            raise ValueError(f"Random.org API error: {data['error']['message']}")
        return np.array(data['result']['random']['data'])

    except (requests.RequestException, ValueError) as e:
        print(f"Error fetching atmospheric noise: {e}. Using fallback Gaussian noise.")
        # fallback: generate noise locally as uniform
        return np.random.uniform(min_val, max_val, size=num_values)


if __name__ == "__main__":
    main()