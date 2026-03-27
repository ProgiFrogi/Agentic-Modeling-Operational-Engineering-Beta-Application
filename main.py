#!/usr/bin/env python3
import sys
import os

# Добавляем корневую директорию в PYTHONPATH
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agents.data_worker import run_data_worker
import pandas as pd
from utils.logger import init_logger

if __name__ == "__main__":
    init_logger()
    df = pd.read_csv("data/train.csv")
    initial_state = {
        "task": "Perform EDA",
        "df": df,
        "data_dir": "data",
        "name_of_file": "train.csv",
        "path_to_comp_desc": "data/KaggleDescription.txt",
        "current_plan": None,
        "analytic_attempts": 0,
        "analytic_max_attempts": 4,
        "satisfy_rate": 0.0,
        "worker_attempts": 0,
        "worker_max_attempts": 3,
        "previous_results": [],
        "done": False,
    }
    result = run_data_worker(initial_state)