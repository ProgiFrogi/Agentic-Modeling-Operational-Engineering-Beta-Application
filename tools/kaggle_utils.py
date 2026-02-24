import datetime
import kaggle

kaggle.api.authenticate()

import os
from kaggle.api.kaggle_api_extended import KaggleApi
import pandas as pd

api = KaggleApi()
api.authenticate()

def get_competition_leaderboard(competition_name, limit=10, output_file=None):
    """
    Получение данных leaderboard указанного соревнования

    Args:
        competition_name (str): Название соревнования на Kaggle
        limit (int): Количество записей для вывода (по умолчанию 10)
        output_file (str): Опциональный путь для сохранения результатов в CSV

    Returns:
        list: Список словарей с данными leaderboard
    """
    leaderboard = api.competition_leaderboard_view(competition_name)
    results = []
    for i, entry in enumerate(leaderboard[:limit]):
        result = {
            'Rank': i,
            'Team Name': entry.team_name,
            'Score': entry.score,
            'LastSubmission': entry.submission_date,
        }
        results.append(result)

    if output_file:
        df = pd.DataFrame(results)
        df.to_csv(output_file, index=False)
        print(f"\nРезультаты сохранены в: {output_file}")

    return results

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


def download_sample_submission(competition_name, searched_file='sample_submition.csv', output_file=None):
    """
    Скачивание файла с соревнования

    Args:
        competition_name (str): Название соревнования
        searched_file(str): название файла для скачивания
        output_file (str): Имя для сохранения файла

    Returns:
        str: Путь к скачанному файлу или None
    """
    try:
        files = api.competition_list_files(competition_name)
        sample_file = None
        files = files.files
        for file in files:
            if searched_file == file.ref.lower():
                sample_file = file
                break

        if not sample_file:
            print(f"❌ Не найден {searched_file}")
            return None
        if not output_file:
            output_file = searched_file

        print(f"\n📥 Скачивание {sample_file.name}...")

        api.competition_download_file(
            competition_name,
            sample_file.name,
            path='./'
        )

        if os.path.exists(sample_file.name):
            os.rename(sample_file.name, output_file)
            print(f"✅ Файл сохранен как: {output_file}")
            return output_file

    except Exception as e:
        print(f"❌ Ошибка скачивания: {str(e)}")
        return None



if __name__ == '__main__':
    print(download_sample_submission("mws-ai-agents-2026", 'test.csv'))