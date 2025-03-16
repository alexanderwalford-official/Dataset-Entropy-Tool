
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
    print("1) Load dataset")
    usr_in = input()
    if usr_in == "1":
        load_csv()
    return


def load_csv():
    dataset_obj = pd.read_csv(DATASET_FILE)
    vals = []
    dt_size = len(dataset_obj)

    print("Dataset size: " + str(dataset_obj.size))

    for _, row in tqdm(dataset_obj.iterrows(), total=dt_size, desc="Processing rows"):
        vals.append(row[COLUMN_NAME])  # Fixed: Correctly access column value

    new_vals = []
    lc = 0

    for i in vals:  # loop through values
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

    # print arrays
    print("Original dataset:")
    print(vals)

    print("New values:")
    print(new_vals)

    # now convert the new_vals into a dataframe

    # save into new csv

    # compare original vals and new vals and generate a difference array

    # save into a new csv

    # perhaps render a line chart containing the original and new values using matplotlib



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