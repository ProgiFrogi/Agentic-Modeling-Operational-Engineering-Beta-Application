"""Kaggle API utilities for competition integration"""

import datetime
import os
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import kaggle
from kaggle.api.kaggle_api_extended import KaggleApi
import pandas as pd

def setup_kaggle_credentials():
    """Set up Kaggle API credentials from token or file"""

    # Способ 1: Использовать токен из переменной окружения
    api_token = os.getenv("KAGGLE_API_TOKEN")
    if api_token:
        os.environ["KAGGLE_API_TOKEN"] = api_token
        return True

    # Способ 2: Использовать файл с токеном
    token_paths = [
        Path.home() / ".kaggle" / "access_token",
        Path("access_token"),
        Path.cwd() / "kaggle_token.txt"
    ]

    for path in token_paths:
        if path.exists():
            with open(path, 'r') as f:
                token = f.read().strip()
            os.environ["KAGGLE_API_TOKEN"] = token
            return True

    # Способ 3: Старый способ с kaggle.json
    kaggle_paths = [
        Path.home() / ".kaggle" / "kaggle.json",
        Path("kaggle.json")
    ]

    for path in kaggle_paths:
        if path.exists():
            with open(path) as f:
                creds = json.load(f)
            os.environ["KAGGLE_USERNAME"] = creds.get("username")
            os.environ["KAGGLE_KEY"] = creds.get("key")
            return True

    print("❌ Kaggle credentials not found!")
    print("Please set KAGGLE_API_TOKEN environment variable or place token in:")
    print("  - ~/.kaggle/access_token")
    print("  - kaggle_token.txt in project directory")
    return False

# Выполняем настройку при импорте
setup_kaggle_credentials()

class KaggleManager:
    """Manager for Kaggle competition operations"""

    def __init__(self, competition_name: str):
        self.competition_name = competition_name

        # Проверяем аутентификацию
        if not (os.getenv("KAGGLE_API_TOKEN") or
                (os.getenv("KAGGLE_USERNAME") and os.getenv("KAGGLE_KEY"))):
            raise Exception("Kaggle API credentials not configured. Please set KAGGLE_API_TOKEN")

        self.api = KaggleApi()
        self.api.authenticate()

    def download_competition_files(self, output_dir: str = "data") -> List[str]:
        """Download all competition files"""
        try:
            print(f"\n📥 Downloading competition files for: {self.competition_name}")

            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            # Download all files
            self.api.competition_download_files(
                self.competition_name,
                path=str(output_path)
            )

            # Unzip files
            import zipfile
            downloaded_files = []
            for zip_file in output_path.glob("*.zip"):
                with zipfile.ZipFile(zip_file, 'r') as zf:
                    zf.extractall(output_path)
                downloaded_files.extend([str(f) for f in output_path.glob("*.csv")])
                zip_file.unlink()

            print(f"✅ Downloaded {len(downloaded_files)} files")
            return downloaded_files

        except Exception as e:
            print(f"❌ Error downloading files: {str(e)}")
            return []

    def get_competition_info(self) -> Dict[str, Any]:
        """Get competition information"""
        try:
            competition = self.api.competition_view(self.competition_name)
            return {
                "title": competition.title,
                "description": competition.description[:500] + "..." if competition.description else "",
                "evaluation_metric": competition.evaluationMetric,
                "reward": competition.reward,
                "deadline": competition.deadline,
                "max_submissions_per_day": competition.maxDailySubmissions,
                "current_leaderboard": self.get_leaderboard(limit=5)
            }
        except Exception as e:
            print(f"❌ Error getting competition info: {str(e)}")
            return {}

    def get_leaderboard(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get competition leaderboard"""
        try:
            leaderboard = self.api.competition_leaderboard_view(self.competition_name)
            results = []
            for i, entry in enumerate(leaderboard[:limit]):
                results.append({
                    'rank': i + 1,
                    'team_name': entry.team_name,
                    'score': entry.score,
                    'last_submission': entry.submission_date,
                })
            return results
        except Exception as e:
            print(f"❌ Error getting leaderboard: {str(e)}")
            return []

    def download_sample_submission(self, output_file: str = "sample_submission.csv") -> Optional[str]:
        """Download sample submission file"""
        try:
            files = self.api.competition_list_files(self.competition_name)
            sample_file = None

            for file in files.files:
                if "sample" in file.ref.lower() or "submission" in file.ref.lower():
                    sample_file = file
                    break

            if not sample_file:
                print(f"❌ Sample submission file not found")
                return None

            print(f"\n📥 Downloading sample submission: {sample_file.name}")
            self.api.competition_download_file(
                self.competition_name,
                sample_file.name,
                path='./'
            )

            if os.path.exists(sample_file.name):
                os.rename(sample_file.name, output_file)
                print(f"✅ Saved as: {output_file}")
                return output_file

        except Exception as e:
            print(f"❌ Error downloading sample: {str(e)}")
            return None

    def submit_prediction(self, submission_file: str, message: Optional[str] = None) -> Tuple[bool, str]:
        """Submit prediction to competition"""
        if not os.path.exists(submission_file):
            return False, f"File not found: {submission_file}"

        try:
            if not message:
                message = f"Submission {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

            print(f"\n📤 Submitting {submission_file} to {self.competition_name}...")
            result = self.api.competition_submit(
                file_name=submission_file,
                message=message,
                competition=self.competition_name
            )
            print(f"✅ Submitted successfully! ID: {result.ref}")
            return True, result.ref

        except Exception as e:
            return False, f"Submission error: {str(e)}"

    def check_submission_status(self, limit: int = 5) -> List[Dict[str, Any]]:
        """Check status of recent submissions"""
        try:
            submissions = self.api.competition_submissions(self.competition_name)
            results = []

            for sub in submissions[:limit]:
                results.append({
                    'id': sub.ref,
                    'date': sub.date,
                    'description': sub.description,
                    'status': sub.status,
                    'public_score': sub.public_score,
                    'private_score': sub.private_score,
                    'error_description': sub.error_description,
                    'url': sub.url
                })
            return results

        except Exception as e:
            print(f"❌ Error checking submissions: {str(e)}")
            return []

    def get_last_submission_score(self) -> Optional[float]:
        """Get score of last submission"""
        submissions = self.check_submission_status(limit=1)
        if submissions and submissions[0]['status'] == 'complete':
            return submissions[0].get('public_score')
        return None

    def analyze_submission_result(self, submission_id: str) -> Dict[str, Any]:
        """Analyze submission result and provide feedback"""
        submissions = self.check_submission_status(limit=10)

        for sub in submissions:
            if sub['id'] == submission_id:
                return {
                    "success": sub['status'] == 'complete',
                    "score": sub.get('public_score'),
                    "error": sub.get('error_description'),
                    "rank": sub.get('rank'),
                    "analysis": self._analyze_score(sub.get('public_score'))
                }

        return {"success": False, "error": "Submission not found"}

    def _analyze_score(self, score: Optional[float]) -> str:
        """Analyze score and provide feedback"""
        if score is None:
            return "Score not available yet"

        leaderboard = self.get_leaderboard(limit=100)
        if leaderboard:
            best_score = leaderboard[0]['score']
            worst_score = leaderboard[-1]['score']

            if score <= best_score:
                return f"Excellent! You're leading with score {score}"
            elif score <= best_score * 1.1:
                return f"Very good! Score {score} is close to leader ({best_score})"
            elif score <= best_score * 1.5:
                return f"Good score {score}, but can be improved. Leader has {best_score}"
            else:
                return f"Score {score} needs improvement. Try different features or models. Leader has {best_score}"

        return f"Score: {score}. Check leaderboard for context"

def extract_json_from_response(response: str) -> Dict[str, Any]:
    """Extract JSON from LLM response"""
    clean_response = response.strip()

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
        import re
        json_match = re.search(r'\{.*\}', clean_response, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass
        return {}