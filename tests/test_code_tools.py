from tools.code_tools import execute_code, save_code, validate_code


def test_validate_code_ok():
    assert validate_code("x = 1\n")["ok"] is True


def test_validate_code_syntax_error():
    r = validate_code("def bad(\n")
    assert r["ok"] is False


def test_execute_code(tmp_path):
    ws = tmp_path / "w"
    ws.mkdir()
    r = execute_code(str(ws), "print('hi')\n", timeout=10)
    assert r["ok"] is True
    assert "hi" in r["stdout"]


def test_execute_code_empty_rejected(tmp_path):
    ws = tmp_path / "w"
    ws.mkdir()
    r = execute_code(str(ws), "   \n", timeout=10)
    assert r["ok"] is False
    assert "empty" in r.get("error", "").lower()


def test_execute_code_syntax_error_skips_subprocess(tmp_path):
    ws = tmp_path / "w"
    ws.mkdir()
    r = execute_code(str(ws), "logging.info(f'broken\n", timeout=10)
    assert r["ok"] is False
    assert "Syntax" in (r.get("stderr") or "") or "syntax" in (r.get("stderr") or "").lower()


def test_execute_code_imports_module_from_data_subdir(tmp_path):
    ws = tmp_path / "w"
    ws.mkdir()
    (ws / "data").mkdir()
    (ws / "data" / "helper.py").write_text("ANSWER = 42\n", encoding="utf-8")
    r = execute_code(str(ws), "from helper import ANSWER\nprint(ANSWER)\n", timeout=10)
    assert r["ok"] is True, r
    assert "42" in r["stdout"]


def test_save_code(tmp_path):
    ws = tmp_path / "w"
    ws.mkdir()
    r = save_code(str(ws), "sub/x.py", "print(1)\n")
    assert r["ok"] is True
    assert (ws / "sub" / "x.py").is_file()
