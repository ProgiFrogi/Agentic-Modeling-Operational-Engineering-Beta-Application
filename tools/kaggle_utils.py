import datetime
import os
import zipfile
from typing import Dict, List, Optional, Tuple

import pandas as pd

_KAGGLE_COMP_URL = "https://www.kaggle.com/competitions/"
_KAGGLE_KERNEL_URL = "https://www.kaggle.com/code/"

_api_instance = None


def _kaggle_api():
    """Lazy Kaggle client so importing this module does not require credentials."""
    global _api_instance
    if _api_instance is None:
        # Kaggle reads KAGGLE_API_TOKEN / KAGGLE_USERNAME+KEY from os.environ.
        # .env is loaded by pydantic-settings into Settings only — not into os.environ —
        # so pull it in here before the first `import kaggle` / authenticate().
        try:
            from dotenv import load_dotenv

            load_dotenv()
        except ImportError:
            pass

        import kaggle
        from kaggle.api.kaggle_api_extended import KaggleApi

        kaggle.api.authenticate()
        _api_instance = KaggleApi()
        _api_instance.authenticate()
    return _api_instance


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
    leaderboard = _kaggle_api().competition_leaderboard_view(competition_name)
    results = []
    for i, entry in enumerate(leaderboard[:limit]):
        result = {
            "Rank": i,
            "Team Name": entry.team_name,
            "Score": entry.score,
            "LastSubmission": entry.submission_date,
        }
        results.append(result)

    if output_file:
        df = pd.DataFrame(results)
        df.to_csv(output_file, index=False)
        print(f"\nРезультаты сохранены в: {output_file}")

    return results


def submit_to_competition(competition_name, submission_file, message=None, quiet: bool = False):
    """
    Отправка решения в соревнование

    Args:
        competition_name (str): Название соревнования
        submission_file (str): Путь к файлу с предсказаниями
        message (str): Комментарий к отправке
        quiet: если True — без print (для агентов / библиотечного вызова)

    Returns:
        str: ID отправки или None
    """
    if not os.path.exists(submission_file):
        if not quiet:
            print(f"❌ Файл не найден: {submission_file}")
        return None

    try:
        if not message:
            message = f"Submission {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}"

        if not quiet:
            print(f"\n📤 Отправка {submission_file} в {competition_name}...")
        result = _kaggle_api().competition_submit(
            file_name=submission_file,
            message=message,
            competition=competition_name,
        )
        if not quiet:
            print(f"✅ Успешно отправлено! ID: {result.ref}, msg: {result.message}")

        return result.ref

    except Exception as e:
        if not quiet:
            print(f"❌ Ошибка отправки: {str(e)}")
        return None


def check_submission_status(competition_name, limit=5, quiet: bool = False):
    """
    Проверка статуса последних отправок

    Args:
        competition_name (str): Название соревнования
        limit (int): Количество последних отправок для показа
        quiet: если True — без print (для агентов)

    Returns:
        list: Список отправок
    """
    try:
        submissions = _kaggle_api().competition_submissions(competition_name)

        if not submissions:
            if not quiet:
                print("📭 Нет отправленных решений")
            return []

        results = []
        for i, sub in enumerate(submissions[:limit], 1):
            submission = {
                "id": sub.ref,
                "date": sub.date,
                "description": sub.description,
                "error_description": sub.error_description,
                "public_score": sub.public_score,
                "private_score": sub.private_score,
                "status": sub.status,
                "submitted_by": sub.submitted_by,
                "submitted_by_ref": sub.submitted_by_ref,
                "url": sub.url,
                "team_name": sub.team_name,
            }
            results.append(submission)
        return results

    except Exception as e:
        if not quiet:
            print(f"❌ Ошибка получения статуса: {str(e)}")
        return []


def download_sample_submission(
    competition_name, searched_file="sample_submission.csv", output_file=None
):
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
        files = _kaggle_api().competition_list_files(competition_name)
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

        _kaggle_api().competition_download_file(
            competition_name,
            sample_file.name,
            path="./",
        )

        if os.path.exists(sample_file.name):
            os.rename(sample_file.name, output_file)
            print(f"✅ Файл сохранен как: {output_file}")
            return output_file

    except Exception as e:
        print(f"❌ Ошибка скачивания: {str(e)}")
        return None


def search_competitions(
    query: Optional[str] = None,
    max_results: int = 20,
    category: Optional[str] = None,
    group: Optional[str] = None,
    sort_by: Optional[str] = None,
) -> List[Dict]:
    """
    Search Kaggle competitions.

    Returns list of dicts: {ref, title, url, reward, teamCount, deadline, category, organizationName}
    """
    results: List[Dict] = []
    page = 1
    page_size = min(20, max(1, max_results))

    try:
        while len(results) < max_results:
            comps = _kaggle_api().competitions_list(
                search=query,
                category=category,
                group=group,
                sort_by=sort_by,
                page=page,
                page_size=page_size,
            )
            if not comps.competitions:
                break
            comps = comps.competitions
            for c in comps:
                results.append(
                    {
                        "ref": getattr(c, "ref", None) or getattr(c, "competition_ref", None),
                        "title": getattr(c, "title", None),
                        "url": _KAGGLE_COMP_URL
                        + (getattr(c, "ref", "") or getattr(c, "competition_ref", "")),
                        "reward": getattr(c, "reward", None),
                        "teamCount": getattr(c, "team_count", None) or getattr(c, "teams", None),
                        "deadline": getattr(c, "deadline", None) or getattr(c, "deadline_utc", None),
                        "category": getattr(c, "category", None),
                        "organizationName": getattr(c, "organization_name", None),
                    }
                )
                if len(results) >= max_results:
                    break
            page += 1
    except Exception as e:
        print(f"search_competitions error: {e}")

    return results[:max_results]


def search_kernels(
    query: Optional[str] = None,
    max_results: int = 20,
    competition: Optional[str] = None,
    language: Optional[str] = "python",
    kernel_type: Optional[str] = None,
    sort_by: Optional[str] = "hotness",
) -> List[Dict]:
    """
    Search Kaggle notebooks (kernels). If competition is provided, restrict to it.

    Returns list of dicts: {ref, title, url, owner, slug, kernelType}
    """
    results: List[Dict] = []
    page = 1
    page_size = min(50, max(1, max_results))
    competition_slug = competition.split("/")[-1] if competition else None

    try:
        while len(results) < max_results:
            kernels = _kaggle_api().kernels_list(
                page=page,
                page_size=page_size,
                search=query,
                competition=competition_slug,
                language=language,
                kernel_type=kernel_type,
                sort_by=sort_by,
            )
            if not kernels:
                break
            for k in kernels:
                author = (
                    getattr(k, "author", None)
                    or getattr(k, "owner_slug", None)
                    or getattr(k, "ownerRef", None)
                )
                slug = getattr(k, "slug", None)
                ref = getattr(k, "ref", None) or (f"{author}/{slug}" if author and slug else None)
                title = getattr(k, "title", None) or slug
                if not ref:
                    continue
                results.append(
                    {
                        "ref": ref,
                        "title": title,
                        "url": _KAGGLE_KERNEL_URL + ref,
                        "author": author,
                        "slug": slug,
                    }
                )
                if len(results) >= max_results:
                    break
            if len(kernels) < page_size:
                break
            page += 1
    except Exception as e:
        print(f"search_kernels error: {e}")

    return results[:max_results]


def download_kernel_notebook(kernel_ref: str, path: str = "./kaggle_notebooks") -> Optional[str]:
    """
    Download a Kaggle kernel (notebook/script) into path and return .ipynb path if found.
    kernel_ref should be in the form 'owner/slug'.
    """
    os.makedirs(path, exist_ok=True)
    try:
        if "/" not in kernel_ref:
            print(f"download_kernel_notebook: invalid ref '{kernel_ref}'")
            return None
        owner, slug = kernel_ref.split("/", 1)
        if os.path.isdir(path):
            path = f"{path}/{owner}_{slug}.ipynb"
            with open(path, "w") as _:
                pass
        _kaggle_api().kernels_pull(kernel=kernel_ref, path=path, metadata=False, quiet=True)
        return path
    except Exception as e:
        print(f"download_kernel_notebook error: {e}")
        return None


def _extract_zip_archives_in_dir(dest_dir: str) -> List[str]:
    """Extract any .zip in dest_dir (Kaggle often delivers a single competition bundle)."""
    extracted: List[str] = []
    for name in os.listdir(dest_dir):
        if not name.lower().endswith(".zip"):
            continue
        zpath = os.path.join(dest_dir, name)
        try:
            with zipfile.ZipFile(zpath, "r") as zf:
                zf.extractall(dest_dir)
            extracted.append(name)
        except zipfile.BadZipFile:
            continue
    return extracted


def setup_competition(competition_ref: str, dest_dir: str) -> Dict:
    """
    Download all competition data files into dest_dir (train/test/sample_submission, etc.).
    Unpacks .zip bundles so CSVs sit under dest_dir (or a subfolder, if the archive has one).
    """
    os.makedirs(dest_dir, exist_ok=True)
    try:
        _kaggle_api().competition_download_files(
            competition_ref, path=dest_dir, quiet=False, force=False
        )
        zips = _extract_zip_archives_in_dir(dest_dir)
        from tools.data_tools import ensure_workspace_kaggle_csv_aliases

        aliases = ensure_workspace_kaggle_csv_aliases(dest_dir)
        return {
            "ok": True,
            "path": dest_dir,
            "competition": competition_ref,
            "extracted_zips": zips,
            "csv_aliases": aliases,
        }
    except Exception as e:
        return {"ok": False, "error": str(e), "competition": competition_ref}


def submit_solution(
    submission_file: str,
    competition_ref: str,
    message: Optional[str] = None,
    quiet: bool = False,
) -> Dict:
    """
    Agent-friendly wrapper around submit_to_competition. Returns a structured dict.
    """
    ref = submit_to_competition(competition_ref, submission_file, message=message, quiet=quiet)
    if ref:
        return {"ok": True, "submission_ref": ref, "file": submission_file}
    return {"ok": False, "error": "submit failed or file missing", "file": submission_file}


if __name__ == "__main__":
    print(download_sample_submission("mws-ai-agents-2026", "sample_submission.csv"))
