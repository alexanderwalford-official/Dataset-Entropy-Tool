import numpy as np
import requests
from scipy.stats import invgamma

def bayesian_std(data, prior=1):
    n = len(data)
    if n < 2:
        return 0 
    
    sample_var = np.var(data, ddof=1)
    posterior = invgamma(n / 2, scale=n * sample_var / 2 + prior)
    return np.sqrt(posterior.mean()) 

def mad_based_std(data):
    med = np.median(data)
    mad = np.median(np.abs(data - med))
    return mad * 1.4826 

def format_float_for_api(min_val, max_val):
    # ensure we don't have identical min and max
    if min_val == max_val:
        raise ValueError("Min and max values must be different.")

    # shift values to make min_val start from 0
    offset = min_val
    range_val = max_val - min_val

    # scale factor: Fit the max value within the allowed range
    max_random_org = 1_000_000_000  # Random.org limit
    factor = max_random_org / range_val  # Compute scaling factor

    # scale min and max to integer values
    min_int = int((min_val - offset) * factor)
    max_int = int((max_val - offset) * factor)

    return min_int, max_int, factor, offset

def randomorg_to_float(random_value, factor, offset):
    return (random_value / factor) + offset 

def fetch_gaussian_noise(num_values, mean, stddev):
    try:
        # generate Gaussian noise
        return np.random.normal(mean, stddev, num_values)
    except Exception as e:
        print(f"Error generating Gaussian noise: {e}.")
        return np.random.normal(0, 1, num_values)  # fallback to standard normal distribution if error occurs
    

def fetch_atmospheric_noise(num_values, min_val, max_val, RANDOM_API_KEY):
    """
    Fetches random generated numbers from random.org.
    Falls back to uniform distribution if the request fails.
    """
    # ensure min_val and max_val are numbers
    try:
        min_val, max_val, factor, offset = format_float_for_api(min_val, max_val)
    except ValueError:
        print(f"[ X ] Invalid range: min_val={min_val}, max_val={max_val}. They must be numeric.")
        return np.random.uniform(0, 1, size=num_values)
    
    if min_val > max_val:
        max_val_n = min_val
        min_val_n = max_val
    else:
        max_val_n = max_val
        min_val_n = min_val

    #print(f"Using Random.org with min={min_val} and max={max_val}")

    url = "https://api.random.org/json-rpc/4/invoke"
    headers = {'Content-Type': 'application/json'}
    payload = {
        "jsonrpc": "2.0",
        "method": "generateIntegers",
        "params": {
            "apiKey": RANDOM_API_KEY,
            "n": num_values,
            "min": min_val_n,
            "max": max_val_n,
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
        print(f"Error fetching atmospheric noise: {e}. Using fallback Gaussian noise. Min: " + str(min_val_n) + ", Max: " + str(max_val_n))
        # fallback: generate noise locally as uniform
        return np.random.uniform(min_val, max_val, size=num_values)
    

def fetch_quantum_entropy(num_values, min_val, max_val, API_KEY):
    """
    Fetches quantum entropy-based random numbers from the ANU Quantum Random Number API.
    Falls back to uniform distribution if the request fails.
    """
    try:
        min_val, max_val, factor, offset = format_float_for_api(min_val, max_val)
    except ValueError:
        print(f"[ X ] Invalid range: min_val={min_val}, max_val={max_val}. They must be numeric.")
        return np.random.uniform(0, 1, size=num_values)
    
    if min_val > max_val:
        min_val, max_val = max_val, min_val
    
    url = "https://api.quantumnumbers.anu.edu.au/"
    headers = {'x-api-key': API_KEY}
    params = {
        "length": num_values,
        "type": "uint8",
        "size": 1
    }
    
    try:
        response = requests.get(url, headers=headers, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        if not data.get("success", False):
            raise ValueError(f"Quantum API error: {data.get('message', 'Unknown error')}")
        
        random_values = np.array(data["data"]) / 255.0  
        scaled_values = min_val + random_values * (max_val - min_val)
        return scaled_values
    
    except (requests.RequestException, ValueError) as e:
        print(f"Error fetching quantum entropy: {e}. Using fallback uniform noise.")
        return np.random.uniform(min_val, max_val, size=num_values)