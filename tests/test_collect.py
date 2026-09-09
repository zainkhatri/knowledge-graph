import os
from mnemosyne.store import Store
from mnemosyne.collect import collect
from mnemosyne import walker

def build_tree(root):
    os.makedirs(os.path.join(root, "projA", "src"))
    open(os.path.join(root, "projA", "README.md"), "w").write("# A")
    open(os.path.join(root, "projA", "src", "main.py"), "w").write("x=1")

def test_collect_stores_nodes_edges_and_understanding(tmp_path):
    build_tree(str(tmp_path))
    st = Store(str(tmp_path / "kg.db"))
    calls = []
    def gen(node, kids):
        calls.append(node["path"]); return f"understood:{node['name']}"
    res = collect(st, str(tmp_path), "ARES", gen=gen)
    assert res["nodes"] >= 3 and res["changed"] == res["nodes"]
    proj = st.get_node("ARES:" + os.path.join(str(tmp_path), "projA"))
    assert proj["understanding"] == "understood:projA"
    kids = st.children("ARES:" + os.path.join(str(tmp_path), "projA"))
    assert any(k["name"] == "src" for k in kids)

def test_collect_is_incremental_on_unchanged(tmp_path):
    build_tree(str(tmp_path))
    st = Store(str(tmp_path / "kg.db"))
    n = {"i": 0}
    def gen(node, kids):
        n["i"] += 1; return "u"
    collect(st, str(tmp_path), "ARES", gen=gen)
    first = n["i"]
    collect(st, str(tmp_path), "ARES", gen=gen)   # nothing changed
    assert n["i"] == first                         # gen not called again

def test_collect_regenerates_changed_folder(tmp_path):
    build_tree(str(tmp_path))
    st = Store(str(tmp_path / "kg.db"))
    def gen(node, kids): return "u"
    collect(st, str(tmp_path), "ARES", gen=gen)
    open(os.path.join(str(tmp_path), "projA", "NEW.txt"), "w").write("z")
    res = collect(st, str(tmp_path), "ARES", gen=gen)
    assert res["changed"] >= 1                      # projA fingerprint changed

def test_collect_vault_not_summarized(tmp_path):
    v = tmp_path / "My Eyes Only"
    (v / "secret").mkdir(parents=True)
    (v / "secret" / "d.txt").write_text("x")
    (tmp_path / "readme.txt").write_text("hi")
    st = Store(str(tmp_path / "kg.db"))
    calls = []
    def gen(node, kids):
        calls.append(node["path"]); return "u"
    collect(st, str(tmp_path), "ARES", vault_pred=walker.default_vault_pred(), gen=gen)
    vnode = st.get_node("ARES:" + str(v))
    assert vnode["kind"] == "vault"
    assert vnode["understanding"] == "Encrypted vault — contents not indexed."
    assert not any("My Eyes Only" in p for p in calls)   # gen never called for the vault
    st.close()
