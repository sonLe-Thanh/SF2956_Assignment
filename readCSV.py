# Read CSV from the files 
import csv
import numpy as np

class CountryInfo:
    def __init__(self, country_name):
        self.name = country_name
        self.data = []

    # A lot of cell has missing values
    # We fill it with the average of the 2 consecutive values
    # As the trend is generally increasing in most of the countris 
    def fill_missing_cells(self):
        for i in range(1, len(self.data) - 1):
            if np.isnan(self.data[i]):
                # If this value is missing
                # Fill the missing one by the average
                # There are cases where one of them is also NaN
                # Then is will be the value of the non-NaN
                # If both of the value are NaN
                # We assign it as 0
                if not np.isnan(self.data[i - 1]) and not np.isnan(self.data[i + 1]):
                    self.data[i] = (self.data[i - 1] + self.data[i + 1]) / 2
                # If both of them are NaN
                elif np.isnan(self.data[i - 1]) and np.isnan(self.data[i + 1]):
                    self.data[i] = 0
                # Here, there can only one of them is zero
                elif np.isnan(self.data[i - 1]):
                    self.data[i] = self.data[i + 1]
                else:
                    self.data[i] = self.data[i - 1]
    
    # Trim both heads
    # As they can also is NaN
    # We will just remove them
    def remove_nan_heads(self):
        non_nan_idx = np.where(~np.isnan(self.data))[0]
        self.data = self.data[non_nan_idx[0] : non_nan_idx[-1] + 1]

    # Reduce the values by a factor
    def rescale_data(self, factor):
        self.data = np.array(self.data)
        self.data /= factor

    # Normalize data
    # As the values of the points can be very large
    # We could normalize them to the range of [0, 1]
    # So that they are more comparable
    def normalize_data(self):
        self.data = np.array(self.data)
        min_val = np.min(self.data)
        max_val = np.max(self.data)
        self.data = (self.data - min_val) / (max_val - min_val)

# Read the csv file and return the list
def readCSV(filepath, exclude_names = []):
    list_country_info = []
    with open(filepath, "r") as file:
        csvread = csv.reader(file) 
        # Skip the header line 
        next(csvread)
        for row in csvread:
            if row[0] not in exclude_names:
                # First element is the country name
                new_country = CountryInfo(row[0])
                # From index 4 to the end is the information
                for i in range(4, 29):
                    new_country.data.append(np.NaN if row[i]=="" else float(row[i]))

                # Add this new country
                list_country_info.append(new_country)

    return list_country_info

# Get key for the sort
def sort_key(country: CountryInfo):
    last_value = country.data[-1]
    # Second to last if possible
    second_to_last_value = country.data[-2] if len(country.data) > 1 else -float("inf")

    if np.isnan(second_to_last_value):
        second_to_last_value = - float("inf")

    # If the last one is NaN, use the second to last for sorting
    if np.isnan(last_value):
        return (- float("inf"), second_to_last_value)
    else:
        return (last_value, second_to_last_value)

# Sort the countries base on the last value of their data,
# If the last one is not available
# The second to last will be used instead
def sort_country(country_list):
    return sorted(country_list, key=sort_key, reverse=True)

# Get top countries
def get_top_k(country_list, top_k):
    return country_list[: top_k]

