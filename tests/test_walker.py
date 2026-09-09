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
