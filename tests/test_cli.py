import os
from mnemosyne import cli

def build_tree(root):
    os.makedirs(os.path.join(root, "projA", "src"))
    open(os.path.join(root, "projA", "README.md"), "w").write("# A")
    open(os.path.join(root, "projA", "src", "main.py"), "w").write("x=1")

def test_cli_reindex_then_search(tmp_path, monkeypatch, capsys):
    build_tree(str(tmp_path))
    monkeypatch.setenv("KG_DB", str(tmp_path / "kg.db"))
    # stub the model so the smoke test needs no Ollama
    from mnemosyne import understanding as U
    monkeypatch.setattr(U, "generate", lambda node, kids=None, http=None: f"desc of {node['name']}")
    cli.main(["reindex", str(tmp_path), "--box", "ARES"])
    cli.main(["search", "projA"])
    out = capsys.readouterr().out
    assert "projA" in out
    cli.main(["stat"])
    assert "nodes" in capsys.readouterr().out
