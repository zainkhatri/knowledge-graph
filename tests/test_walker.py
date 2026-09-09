import os
from mnemosyne import walker

def build_tree(root):
    # root/
    #   projA/ (has README -> project, branch)
    #     src/  (small -> folder)
    #   photos/ (60 jpgs, no subdirs -> file-cluster, no recurse)
    os.makedirs(os.path.join(root, "projA", "src"))
    open(os.path.join(root, "projA", "README.md"), "w").write("# A")
    open(os.path.join(root, "projA", "src", "main.py"), "w").write("x=1")
    os.makedirs(os.path.join(root, "photos"))
    for i in range(60):
        open(os.path.join(root, "photos", f"img{i}.jpg"), "w").write("x")

def test_classify(tmp_path):
    build_tree(str(tmp_path))
    assert walker.classify(str(tmp_path / "projA"), ["src"], ["README.md"]) == ("branch", "project")
    assert walker.classify(str(tmp_path / "photos"), [], [f"img{i}.jpg" for i in range(60)]) == ("cluster", "file-cluster")

def test_walk_adaptive_depth(tmp_path):
    build_tree(str(tmp_path))
    nodes = list(walker.walk(str(tmp_path), "ARES"))
    by_path = {n["path"]: n for n in nodes}
    # projA and its src are nodes; photos is one file-cluster node; no per-image nodes
    assert os.path.join(str(tmp_path), "projA") in by_path
    assert os.path.join(str(tmp_path), "projA", "src") in by_path
    photos = by_path[os.path.join(str(tmp_path), "photos")]
    assert photos["kind"] == "file-cluster"
    assert not any("img0.jpg" in p for p in by_path)          # leaves not walked
    assert photos["meta"]["n_files"] == 60
    # ids and parent links
    assert by_path[os.path.join(str(tmp_path), "projA")]["id"] == "ARES:" + os.path.join(str(tmp_path), "projA")
    assert by_path[os.path.join(str(tmp_path), "projA", "src")]["parent"] == "ARES:" + os.path.join(str(tmp_path), "projA")

def test_vault_is_name_only_and_not_descended(tmp_path):
    import os
    vault = tmp_path / "My Eyes Only"
    (vault / "secret").mkdir(parents=True)
    (vault / "secret" / "diary.txt").write_text("private")
    from mnemosyne import walker
    nodes = list(walker.walk(str(tmp_path), "ARES", vault_pred=walker.default_vault_pred()))
    paths = {n["path"]: n for n in nodes}
    v = paths[str(vault)]
    assert v["kind"] == "vault"
    assert v["understanding"] == "Encrypted vault — contents not indexed."
    # nothing under the vault was walked
    assert not any("secret" in p for p in paths)


def test_walk_skips_ignore_dirs(tmp_path):
    import os
    from mnemosyne import walker
    os.makedirs(os.path.join(str(tmp_path), "proj", "node_modules", "left-pad"))
    os.makedirs(os.path.join(str(tmp_path), "proj", "src"))
    open(os.path.join(str(tmp_path), "proj", "package.json"), "w").write("{}")
    nodes = list(walker.walk(str(tmp_path), "ARES"))
    paths = [n["path"] for n in nodes]
    assert any(p.endswith(os.sep + "src") for p in paths)
    assert not any("node_modules" in p for p in paths)
