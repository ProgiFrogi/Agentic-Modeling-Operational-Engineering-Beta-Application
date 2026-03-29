# prompts/coder_prompts.py
INITIAL_CODE_PROMPT = """
Write Python code for the following task:
{task}

Important guidelines:
1. Use proper spacing and formatting
2. Add spaces after commas: "func(arg1, arg2)" not "func(arg1,arg2)"
3. Put each statement on a new line
4. Use proper indentation (4 spaces per level)
5. Ensure the code runs without errors
6. If using sklearn OneHotEncoder, use sparse_output=False
7. Use .ravel() when assigning imputed values
8. Don't use matplotlib or seaborn for visualizations

Memory efficiency for large datasets:
1. Work with large datasets efficiently
2. DO NOT create explosion of features - avoid OneHotEncoder on high-cardinality columns
3. For columns with > 50 unique values, use:
   - Label Encoding, or
   - Frequency encoding, or
   - Keep as-is
4. Always print shapes and memory usage after operations
5. Handle missing values:
   - Numerical: fill with median/mean
   - Categorical: fill with "Unknown" or mode
6. For datetime: extract useful features (year, month, day, dayofweek)
7. Use StandardScaler for numerical features
8. Save processed files with .to_csv(index=False)
9. Print progress messages

{extra_rules}

Output only the code, no explanations.
"""

FIX_CODE_PROMPT = """
The following code had an error:

{code}

Error: {error}

Please provide corrected code. Output only the code, no explanations.
"""

TRAINING_CODE_PROMPT = """
Write Python code to train a model on the data.

IMPORTANT: You are ALREADY in the session directory. Use relative paths.

CRITICAL INFORMATION:
- Target column name is: '{target_column}'
- Task type: {problem_type}
- Evaluation metric: {metric}

Requirements:
1. Load data from: 'train.csv' and 'test.csv' (use relative paths)
2. Target column: '{target_column}'
3. Configuration: {config}
4. Save trained model to: 'model.pkl' (use joblib.dump)
5. Save predictions on test data to: 'predictions.csv'
6. Calculate and print model score on validation set
7. Print feature importance if applicable

Rules:
- Use memory-efficient code
- Use pandas fillna() for missing values
- For categorical columns with >50 unique values, use frequency encoding
- For categorical columns with <=50 unique values, use LabelEncoder
- Scale numerical features with StandardScaler
- Use train/val split (80/20)
- Save model with joblib.dump
- Print all metrics
- Save predictions with format: 'index,prediction'

Output ONLY the Python code, no explanations.
"""