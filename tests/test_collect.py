import os
from atlas.store import Store
from atlas.collect import collect
from atlas import walker

def build_tree(root):
    os.makedirs(os.path.join(root, "projA", "src"))
    open(os.path.join(root, "projA", "README.md"), "w").write("# A")
    open(os.path.join(root, "projA", "src", "main.py"), "w").write("x=1")

# The DB must live OUTSIDE the walked tree (as a sibling of "root", not inside
# it) — SQLite's WAL/SHM files are created as a side effect of the first write,
# and if they land inside the walked directory, that directory's OWN fingerprint
# (which hashes its immediate children's name/mtime/size) changes between the
# first and second collect() call purely from the database's own bookkeeping
# files appearing, with zero actual content change. That made
# test_collect_is_incremental_on_unchanged fail nondeterministically depending
# on whether WAL files had already appeared by the time of the first walk.

def test_collect_stores_nodes_edges_and_understanding(tmp_path):
    root = str(tmp_path / "root")
    os.makedirs(root)
    build_tree(root)
    st = Store(str(tmp_path / "kg.db"))
    calls = []
    def gen(node, kids):
        calls.append(node["path"]); return f"understood:{node['name']}"
    res = collect(st, root, "ARES", gen=gen)
    assert res["nodes"] >= 3 and res["changed"] == res["nodes"]
    proj = st.get_node("ARES:" + os.path.join(root, "projA"))
    assert proj["understanding"] == "understood:projA"
    kids = st.children("ARES:" + os.path.join(root, "projA"))
    assert any(k["name"] == "src" for k in kids)

def test_collect_is_incremental_on_unchanged(tmp_path):
    root = str(tmp_path / "root")
    os.makedirs(root)
    build_tree(root)
    st = Store(str(tmp_path / "kg.db"))
    n = {"i": 0}
    def gen(node, kids):
        n["i"] += 1; return "u"
    collect(st, root, "ARES", gen=gen)
    first = n["i"]
    collect(st, root, "ARES", gen=gen)   # nothing changed
    assert n["i"] == first                # gen not called again

def test_collect_regenerates_changed_folder(tmp_path):
    root = str(tmp_path / "root")
    os.makedirs(root)
    build_tree(root)
    st = Store(str(tmp_path / "kg.db"))
    def gen(node, kids): return "u"
    collect(st, root, "ARES", gen=gen)
    open(os.path.join(root, "projA", "NEW.txt"), "w").write("z")
    res = collect(st, root, "ARES", gen=gen)
    assert res["changed"] >= 1                      # projA fingerprint changed

def test_collect_vault_not_summarized(tmp_path):
    root = str(tmp_path / "root")
    os.makedirs(root)
    v = tmp_path / "root" / "My Eyes Only"
    (v / "secret").mkdir(parents=True)
    (v / "secret" / "d.txt").write_text("x")
    (tmp_path / "root" / "readme.txt").write_text("hi")
    st = Store(str(tmp_path / "kg.db"))
    calls = []
    def gen(node, kids):
        calls.append(node["path"]); return "u"
    collect(st, root, "ARES", vault_pred=walker.default_vault_pred(), gen=gen)
    vnode = st.get_node("ARES:" + str(v))
    assert vnode["kind"] == "vault"
    assert vnode["understanding"] == "Encrypted vault — contents not indexed."
    assert not any("My Eyes Only" in p for p in calls)   # gen never called for the vault
    st.close()

def test_collect_stores_embedding_alongside_understanding(tmp_path):
    root = str(tmp_path / "root")
    os.makedirs(root)
    build_tree(root)
    st = Store(str(tmp_path / "kg.db"))
    def gen(node, kids): return f"understood:{node['name']}"
    def embed_fn(text, http=None): return [1.0, 2.0]
    collect(st, root, "ARES", gen=gen, embed_fn=embed_fn)
    proj = st.get_node("ARES:" + os.path.join(root, "projA"))
    assert st.blob_to_vec(proj["embedding"]).tolist() == [1.0, 2.0]

def test_collect_reuses_cached_embedding_when_unchanged(tmp_path):
    root = str(tmp_path / "root")
    os.makedirs(root)
    build_tree(root)
    st = Store(str(tmp_path / "kg.db"))
    calls = {"n": 0}
    def gen(node, kids): return "u"
    def embed_fn(text, http=None):
        calls["n"] += 1; return [float(calls["n"])]
    collect(st, root, "ARES", gen=gen, embed_fn=embed_fn)
    first_calls = calls["n"]
    collect(st, root, "ARES", gen=gen, embed_fn=embed_fn)   # nothing changed
    assert calls["n"] == first_calls   # embed_fn not called again, cache reused

def test_collect_vault_nodes_never_embedded(tmp_path):
    root = str(tmp_path / "root")
    os.makedirs(root)
    v = tmp_path / "root" / "My Eyes Only"
    (v / "secret").mkdir(parents=True)
    (v / "secret" / "d.txt").write_text("x")
    (tmp_path / "root" / "readme.txt").write_text("hi")
    st = Store(str(tmp_path / "kg.db"))
    from atlas import walker
    def gen(node, kids): return "u"
    def embed_fn(text, http=None): return [1.0]
    collect(st, root, "ARES", vault_pred=walker.default_vault_pred(), gen=gen, embed_fn=embed_fn)
    vnode = st.get_node("ARES:" + str(v))
    assert vnode["embedding"] is None
    st.close()
