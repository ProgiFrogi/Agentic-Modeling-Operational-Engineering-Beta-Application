import datetime
import os
from kaggle.api.kaggle_api_extended import KaggleApi

api = KaggleApi()
api.authenticate()

def submit_to_competition(competition_name, submission_file, message=None):
    """
    Отправка решения в соревнование

    Args:
        competition_name (str): Название соревнования
        submission_file (str): Путь к файлу с предсказаниями
        message (str): Комментарий к отправке

    Returns:
        str: ID отправки или None
    """
    if not os.path.exists(submission_file):
        print(f"❌ Файл не найден: {submission_file}")
        return None

    try:
        if not message:
            message = f"Submission {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}"

        print(f"\n📤 Отправка {submission_file} в {competition_name}...")
        result = api.competition_submit(
            file_name=submission_file,
            message=message,
            competition=competition_name
        )
        print(f"✅ Успешно отправлено! ID: {result.ref}, msg: {result.message}")

        return result.ref

    except Exception as e:
        print(f"❌ Ошибка отправки: {str(e)}")
        return None


def check_submission_status(competition_name, limit=5):
    """
    Проверка статуса последних отправок

    Args:
        competition_name (str): Название соревнования
        limit (int): Количество последних отправок для показа

    Returns:
        list: Список отправок
    """
    try:
        submissions = api.competition_submissions(competition_name)

        if not submissions:
            print("📭 Нет отправленных решений")
            return []

        results = []
        for i, sub in enumerate(submissions[:limit], 1):
            submission = {
                'id': sub.ref,
                'date': sub.date,
                'description': sub.description,
                'error_description': sub.error_description,
                'public_score': sub.public_score,
                'private_score': sub.private_score,
                'status': sub.status,
                'submitted_by': sub.submitted_by,
                'submitted_by_ref': sub.submitted_by_ref,
                'url': sub.url,
                'team_name': sub.team_name,
            }
            results.append(submission)
        return results

    except Exception as e:
        print(f"❌ Ошибка получения статуса: {str(e)}")
        return []
