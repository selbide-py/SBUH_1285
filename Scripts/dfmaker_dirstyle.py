import os
import pandas as pd
import glob

folder_path = '/content/dataset/IN-Abs/train-data'

def get_judgement_and_summary_file_paths(folder_path):

 judgement_file_path = glob.glob(folder_path + 'judgement/*.txt')[0]
 summary_file_path = glob.glob(folder_path + 'summary/*.txt')[0]

 return (judgement_file_path, summary_file_path)


judgement_folder_paths = glob.glob('judgement/*')
summary_folder_paths = glob.glob('summary/*')

all_file_paths = []

for judgement_folder_path in judgement_folder_paths:
 summary_folder_path = judgement_folder_path.replace('judgement', 'summary')

 judgement_file_path, summary_file_path = get_judgement_and_summary_file_paths(judgement_folder_path)

 all_file_paths.append((judgement_file_path, summary_file_path))

dataframe = pd.DataFrame(all_file_paths, columns=['judgement_file_path', 'summary_file_path'])
print(dataframe)

duplicates = dataframe['judgement_file_path'].duplicated()

dataframe = dataframe[~duplicates] # drop duplicatse

dataframe = pd.DataFrame(all_file_paths, columns=['judgement_file_path', 'summary_file_path'])

dataframe = pd.merge(dataframe[['judgement_file_path']], dataframe[['summary_file_path']], on='judgement_file_path', how='inner')

dataframe['judgement_index'] = dataframe['judgement_file_path'].index
dataframe['summary_index'] = dataframe['summary_file_path'].index

dataframe = pd.merge(dataframe[['judgement_index', 'judgement_file_path']], dataframe[['summary_index', 'summary_file_path']], on='judgement_index', how='inner')

dataframe = dataframe[['judgement_file_path', 'summary_file_path']].apply(pd.read_csv)

print(dataframe)
