import pandas as pd
import re
import os

# Specify the folder path
folder_path = r"D:\Downloads\dataset\IN-Abs\train-data\judgement_small"

# List of specific files you want to read
specific_files = ["3.txt", "1.txt"]
texts = []

# Lists to store extracted data
case_identifiers = []
case_descriptions = []
lawyer_representations = []
judgment_dates = []
presiding_judges = []
case_no = []

# Read each specific file


def work0():
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        # Make sure it's a file, not a subdirectory or other type of file
        if os.path.isfile(file_path):
            with open(file_path, 'r', encoding='utf-8') as file:
                l1 = file.readline()
                case_identifiers.append(l1)
                case_no.append(file_name)
                # content = file.read()
                print(file_path)


def work1():
    for file_name in specific_files:
        if file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)
            # Make sure it's a file, not a subdirectory or other type of file
            if os.path.isfile(file_path):
                with open(file_path, 'r', encoding='utf-8') as file:
                    l1 = file.readline()
                    case_identifiers.append(l1)
                    # content = file.read()
                    print(file_path)

        # # Check if the file exists
        # if os.path.exists(file_path) and os.path.isfile(file_path):
        #     with open(file_path, 'r') as file:
        #         l1 = file.readline()
        #         case_identifiers.append(l1)
        #         content = file.read()
        #         print("File worked")
        #         texts.append(content)

        #         # print(f"Contents of {file_name}:\n{content}\n{'-'*40}\n")
        # else:
        #     print(f"{file_name} not found in the specified folder.\n")


def work2():
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


fields = ["Case Identifier", "Case Descirption",
          "Lawyer Representations", "Judgement Date", "Presiding Judge", ""]


def work3():
    for file_name in specific_files:
        file_path = os.path.join(folder_path, file_name)
        if os.path.exists(file_path) and os.path.isfile(file_path):
            with open(file_path, 'r') as file:
                for i in range(7):
                    None
                    # make a dict of all the fields, and a list of the new enteries appending them, and then run a single loop
                    # to club the iterations together

        print("Iteration Done")


# __main__
work0()


# Creating a dataframe
df = pd.DataFrame({
    'File Name': case_no,
    'Case Identifier': case_identifiers,
    # 'Case Description': case_descriptions,
    # 'Lawyer Representations': lawyer_representations,
    # 'Judgment Date': judgment_dates,
    # 'Presiding Judge': presiding_judges
})

print(df)

# # Write dataframe to CSV
df.to_csv('output.csv', sep=",", index=False)

"""
# TODO

format to be followed to read the files
lines:
1 - Civil Appeal/ Appeal (no. x of y)
2 - What's the appeal, under what section, date mentioned, so are the laws cited if any
2 - Under which court (location) and under which Judge, and any cases refered to 
3 - Name of lawyer (who with) [apellant]
4 - Respondent
5 - Date
6 - Judgement delivered by x [Name of the judge]
7 - Cause for appeal
"""

# TODO Note to self, we have to make a reinforcement training based model that we can work on the train dataset to get the output as close to the test dataset as possible
# Now I am thinking, how do i get lora involved, I need to understand what the fuck reinforcment does, then I have to clean out the data and put into a dataset and then we'll
# see from there, sounds good
