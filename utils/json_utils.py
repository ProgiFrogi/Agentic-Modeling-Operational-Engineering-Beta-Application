import json
from typing import Dict, Any


def extract_json_from_response(response: str) -> Dict[str, Any]:
    """Extract JSON from LLM response, removing markdown formatting"""
    clean_response = response.strip()

    # Remove markdown code blocks if present
    if clean_response.startswith("```json"):
        clean_response = clean_response.split("```json")[1]
    elif clean_response.startswith("```"):
        clean_response = clean_response.split("```")[1]

    if clean_response.endswith("```"):
        clean_response = clean_response.rsplit("```", 1)[0]

    clean_response = clean_response.strip()

    try:
        return json.loads(clean_response)
    except json.JSONDecodeError:
        # Try to find JSON object using regex
        import re
        json_match = re.search(r'\{.*\}', clean_response, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass
        return {}
