import pandas as pd
import re
import os

# Specify the folder path
folder_path = r"D:\Downloads\dataset\IN-Abs\train-data\judgement"

# List of specific files you want to read
specific_files = ["3.txt", "1.txt"]
texts = []

# Lists to store extracted data
case_identifiers = []
case_descriptions = []
lawyer_representations = []
judgment_dates = []
presiding_judges = []

# Read each specific file
for file_name in specific_files:
    file_path = os.path.join(folder_path, file_name)

    # Check if the file exists
    if os.path.exists(file_path) and os.path.isfile(file_path):
        with open(file_path, 'r') as file:
            l1 = file.readline()
            case_identifiers.append(l1)
            content = file.read()
            print("File worked")
            texts.append(content)

            # print(f"Contents of {file_name}:\n{content}\n{'-'*40}\n")
    else:
        print(f"{file_name} not found in the specified folder.\n")

for text in texts:
    # Extracting case identifier
    # case_id_match = re.search(r'^[A-Za-z\s\.\-]+\d+ of \d+\.', text)
    # case_id_match = read_specific_line(text, 0)
    # print(case_id_match)
    # case_identifiers.append(case_id_match.group() if case_id_match else None)

    # Extracting case description or brief statement
    case_desc_match = re.search(r'(?<=\d{4}\.)\s+([\w\s]+)', text)
    case_descriptions.append(case_desc_match.group(1)
                             if case_desc_match else None)

    # Extracting lawyer representations
    lawyers_match = re.findall(
        r'([A-Z][\w\s\.]+)(\([\w\s,\.]+\))? for the (appellant|petitioner|respondent|opposite party)', text)
    lawyer_representations.append(lawyers_match if lawyers_match else None)

    # Extracting judgment date
    date_match = re.search(r'\d{4}\.\s+\w+\s+\d+\.', text)
    judgment_dates.append(date_match.group() if date_match else None)

    # Extracting presiding judges
    judge_match = re.search(
        r'(?:The Judgment of the Court was delivered by|judgments were delivered:)\s+([\w\s\.]+)', text)
    presiding_judges.append(judge_match.group(1) if judge_match else None)

# Creating a dataframe
df = pd.DataFrame({
    'Case Identifier': case_identifiers,
    'Case Description': case_descriptions,
    'Lawyer Representations': lawyer_representations,
    'Judgment Date': judgment_dates,
    'Presiding Judge': presiding_judges
})

print(df)
