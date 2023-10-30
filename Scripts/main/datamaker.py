import os

import os
import pandas as pd

folder_path = "/path/to/your/folder"

# Create an empty DataFrame
df = pd.DataFrame()

# Iterate through the files in the directory
for filename in os.listdir(folder_path):
    if filename.endswith(".txt"):
        file_path = os.path.join(folder_path, filename)
        with open(file_path, 'r') as f:
            # Assuming each .txt file contains a single column of data
            data = pd.read_csv(f, header=None, sep="\n", names=[filename])
            df = pd.concat([df, data], axis=1)  # Concatenate the data horizontally

# Save the DataFrame to a .csv file
df.to_csv("/path/to/save/final.csv", index=False)

"""
import os
import pandas as pd

folder_path = "/path/to/your/folder"

# Create an empty DataFrame with specified columns
df = pd.DataFrame(columns=["Name", "Age", "City"])

# Iterate through the files in the directory
for filename in os.listdir(folder_path):
    if filename.endswith(".txt"):
        file_path = os.path.join(folder_path, filename)
        with open(file_path, 'r') as f:
            content = f.readlines()
            # Remove any newlines or whitespace
            content = [x.strip() for x in content if x.strip() != '']
            data_dict = {}
            for line in content:
                key, value = line.split(":")
                data_dict[key.strip()] = value.strip()
            df = df.append(data_dict, ignore_index=True)

# Save the DataFrame to a .csv file
df.to_csv("/path/to/save/final.csv", index=False)

# TODO
 
format to be followed to read the files
lines:
1 - name of the case (?)
2 - Application / article / what for
3 - facts ?
4 - 
"""