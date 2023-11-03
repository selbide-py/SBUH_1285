import os
import pandas as pd

file_path = '/content/dataset/IN-Abs/train-data'

df = pd.DataFrame(columns=["judgement", "summary"])

# Define a function to read text from a text file
def read_text_from_file(file_path):
  judgements = []  # List to store judgments
  summaries = []   # List to store summaries

    # Read judgment files
  with os.scandir(file_path+'/judgement') as entries:
    # Iterate over the first 2500 items in the directory.
    for i in range(2500):
      entry = next(entries)
      if entry.is_file() and entry.name.endswith(".txt"):
        with open(entry.path, 'r', encoding='utf-8') as file:
          jud = file.read()
          judgements.append(jud)


   # Read summary files
  with os.scandir(file_path+'/summary') as entries:
    for i in range(2500):
      entry = next(entries)
      if entry.is_file() and entry.name.endswith(".txt"):
        with open(entry.path, 'r', encoding='utf-8') as file:
          sums = file.read()
          summaries.append(sums)

  return judgements, summaries

# Call the function to read text from files
judgements, summaries = read_text_from_file(file_path)

# Add the data to the DataFrame
df["judgement"] = judgements
df["summary"] = summaries


print (df.head())
print (df.shape)