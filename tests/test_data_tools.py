import pandas as pd

from tools.data_tools import (
    check_file,
    ensure_workspace_kaggle_csv_aliases,
    inspect_data,
    list_workspace_files,
    read_data,
)


def test_check_file_and_read(tmp_path):
    root = tmp_path / "ws"
    root.mkdir()
    csv = root / "train.csv"
    pd.DataFrame({"a": [1, 2], "b": [3, 4]}).to_csv(csv, index=False)

    assert check_file(str(root), "train.csv")["exists"] is True
    r = read_data(str(root), "train.csv", nrows=10)
    assert r["ok"] is True
    assert "a" in r["columns"]


def test_path_escape_rejected(tmp_path):
    root = tmp_path / "ws"
    root.mkdir()
    bad = check_file(str(root), "../outside")
    assert bad["exists"] is False
    assert "error" in bad


def test_read_data_rewrites_data_train_to_root(tmp_path):
    root = tmp_path / "ws"
    root.mkdir()
    pd.DataFrame({"a": [1]}).to_csv(root / "train.csv", index=False)
    r = read_data(str(root), "data/train.csv", nrows=10)
    assert r["ok"] is True
    assert r["path"].endswith("train.csv")


def test_ensure_workspace_kaggle_csv_aliases(tmp_path):
    root = tmp_path / "ws"
    root.mkdir()
    pd.DataFrame({"x": [1]}).to_csv(root / "train.csv", index=False)
    created = ensure_workspace_kaggle_csv_aliases(str(root))
    assert len(created) >= 1
    assert (root / "data" / "train.csv").exists()


def test_list_workspace_files_skips_pycache(tmp_path):
    root = tmp_path / "ws"
    root.mkdir()
    (root / "train.csv").write_text("a\n1\n", encoding="utf-8")
    d = root / "data"
    d.mkdir()
    (d / "x.csv").write_text("b\n", encoding="utf-8")
    cache = d / "__pycache__"
    cache.mkdir()
    (cache / "junk.pyc").write_bytes(b"x")
    r = list_workspace_files(str(root), glob_pattern="*.csv")
    assert r["ok"] is True
    names = {Path(p).name for p in r["files"]}
    assert "train.csv" in names
    assert "x.csv" in names
    assert "junk.pyc" not in " ".join(r["files"])


def test_inspect_data(tmp_path):
    root = tmp_path / "ws"
    root.mkdir()
    pd.DataFrame({"x": [1.0, None]}).to_csv(root / "t.csv", index=False)
    info = inspect_data(str(root), "t.csv")
    assert info["ok"] is True
    assert "null_counts" in info
