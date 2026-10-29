import pandas as pd

def load_data(file_path):
    students_df_raw = pd.read_csv(file_path, sep=",")
    return students_df_raw