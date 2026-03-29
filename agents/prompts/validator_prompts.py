# prompts/validator_prompts.py
VALIDATION_PROMPT = """
You are a model validator. Evaluate the model performance:

Metric: {metric}
Score: {score}
Threshold: {threshold}
Data shape: {n_samples} validation samples

Generate a validation report with:
1. Is the model acceptable? (score <= threshold for error metrics, >= threshold for R2)
2. Recommendations for improvement if needed
3. Brief analysis of the result

Output in JSON format:
{{
    "passed": true/false,
    "analysis": "brief analysis",
    "recommendations": ["recommendation1", "recommendation2"]
}}
"""