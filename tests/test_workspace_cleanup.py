from pathlib import Path

import pandas as pd

from tools.workspace_cleanup import clean_agent_artifacts, fork_workspace_with_data_symlinks


def test_clean_agent_artifacts(tmp_path):
    root = tmp_path / "ws"
    root.mkdir()
    scripts = root / "scripts"
    scripts.mkdir()
    (scripts / "_agent_deadbeef.py").write_text("x", encoding="utf-8")
    (root / "tmpabc123.py").write_text("y", encoding="utf-8")
    cache = scripts / "__pycache__"
    cache.mkdir()
    (cache / "x.pyc").write_bytes(b"\0")

    r = clean_agent_artifacts(str(root))
    assert r["ok"] is True
    assert not (scripts / "_agent_deadbeef.py").exists()
    assert not (root / "tmpabc123.py").exists()
    assert not cache.exists()


def test_fork_workspace_with_data_symlinks(tmp_path):
    base = tmp_path / "mws-ai-agents-2026"
    base.mkdir()
    pd.DataFrame({"a": [1]}).to_csv(base / "train.csv", index=False)
    dest = Path(fork_workspace_with_data_symlinks(str(base)))
    assert dest.is_dir()
    assert dest.name.startswith("mws-ai-agents-2026_")
    assert (dest / "train.csv").is_symlink() or (dest / "train.csv").is_file()
