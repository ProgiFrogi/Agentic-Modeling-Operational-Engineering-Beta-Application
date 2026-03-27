import io
import sys

import pandas as pd


def get_initial_data_form_df(df: pd.DataFrame, desc_of_competition: str, name_of_file: str) -> dict:
    result = {
        "desc_of_columns": None,
        "df_info_result": None,
        "first_dataset_string": None,
        "desc_of_data_size": None,
        "history": None,
        "name_of_file": None,
    }
    with open(desc_of_competition, "r") as f:
        result["desc_of_columns"] = f.read()

    buffer = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = buffer

    try:
        df.info()
        info_str = buffer.getvalue()
    finally:
        sys.stdout = old_stdout
        buffer.close()
    result["df_info_result"] = info_str
    result["first_dataset_string"] = df.head()
    result["desc_of_data_size"] = len(df)
    result["name_of_file"] = name_of_file

    return result
