import os
from atlas.store import Store
from atlas.prune import prune_excluded


def _node(st, nid, kind, path, text="some understanding text here"):
    st.upsert_node({"id": nid, "box": "ARES", "kind": kind, "path": path, "name": os.path.basename(path),
                    "understanding": text})


def test_prune_removes_excluded_files_and_folders_but_keeps_chats(tmp_path):
    st = Store(str(tmp_path / "kg.db"))
    arch = "/pool/PERSONAL/CLAUDE-CODE-SESSIONS/ARES/p/s1"
    _node(st, "ARES:" + arch, "folder", arch)
    _node(st, "ARES:" + arch + "/tool-results/b.txt", "file-content", arch + "/tool-results/b.txt")
    _node(st, "ARES:/pool/.claude/skills/x.md", "file-content", "/pool/.claude/skills/x.md")
    _node(st, "ARES:/pool/PROJECTS/app", "folder", "/pool/PROJECTS/app")
    _node(st, "ARES:chat/s1", "chat", arch + "/s1.jsonl")
    st.add_edge("ARES:" + arch, "ARES:" + arch + "/tool-results/b.txt", "contains")

    r = prune_excluded(st)

    assert r["pruned"] == 3
    assert st.get_node("ARES:" + arch) is None
    assert st.get_node("ARES:/pool/.claude/skills/x.md") is None
    assert st.get_node("ARES:/pool/PROJECTS/app") is not None
    assert st.get_node("ARES:chat/s1") is not None       # chats live in the archive by design
    assert st.search("understanding text") and all(n["kind"] != "file-content" for n in st.search("understanding text"))
    assert st.db.execute("SELECT count(*) FROM edges").fetchone()[0] == 0
    st.close()


def test_prune_marks_near_empty_content_and_drops_it_from_search(tmp_path):
    st = Store(str(tmp_path / "kg.db"))
    _node(st, "ARES:/pool/PHOTOS/a.jpg", "file-content", "/pool/PHOTOS/a.jpg", text="a i <> x")
    _node(st, "ARES:/pool/PHOTOS/b.jpg", "file-content", "/pool/PHOTOS/b.jpg", text="zebra crossing sign downtown")

    r = prune_excluded(st)

    assert r["emptied"] == 1
    assert st.get_node("ARES:/pool/PHOTOS/a.jpg")["status"] == "empty"
    assert [n["name"] for n in st.search("zebra")] == ["b.jpg"]
    assert st.db.execute("SELECT count(*) FROM nodes_fts WHERE id='ARES:/pool/PHOTOS/a.jpg'").fetchone()[0] == 0
    st.close()
