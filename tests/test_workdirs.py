from atlas.store import Store
from atlas.workdirs import canonical, link_workdirs


def test_canonical_paths():
    u = "/srv/dev-disk-by-uuid-27b8f17b-bc24-456f-852c-212358ed968e"
    assert canonical("ZEUS", u + "/BUSINESS/FAI") == ("ZEUS", "/srv/mergerfs/PROMETHEUS/BUSINESS/FAI")
    assert canonical("ZEUS", u) == ("ZEUS", "/srv/mergerfs/PROMETHEUS")
    assert canonical("MAC", "/Users/zk/PROMETHEUS/ARES/PROJECTS/x") == ("ARES", "/mnt/nvme/PROMETHEUS/PROJECTS/x")
    assert canonical("MAC", "/Users/zk/PROMETHEUS/.ARES.a/PROJECTS/x") == ("ARES", "/mnt/nvme/PROMETHEUS/PROJECTS/x")
    assert canonical("MAC", "/Users/zk/PROMETHEUS/.ZEUS.a/BUSINESS") == ("ZEUS", "/srv/mergerfs/PROMETHEUS/BUSINESS")
    assert canonical("MAC", "/Volumes/PROMETHEUS/WORK") == ("MAC", "/Volumes/PROMETHEUS/WORK")   # ambiguous: stay put
    assert canonical("ARES", "/mnt/nvme/PROMETHEUS/x") == ("ARES", "/mnt/nvme/PROMETHEUS/x")


def _chat(st, box, sid, cwd, name="2026-09-01 · did a thing"):
    st.upsert_node({"id": f"{box}:chat/{sid}", "box": box, "kind": "chat", "path": f"/a/{sid}",
                    "name": name, "understanding": "u", "mtime": 1, "meta": {"session": sid, "cwd": cwd}})
    st.add_edge(f"{box}:chats", f"{box}:chat/{sid}", "contains")


def _parents(st, nid):
    return {r[0] for r in st.db.execute("SELECT src FROM edges WHERE dst=? AND type='contains'", (nid,))}


def test_mac_session_on_ares_mount_links_to_real_ares_folder(tmp_path):
    st = Store(str(tmp_path / "kg.db"))
    st.upsert_node({"id": "ARES:/mnt/nvme/PROMETHEUS/PROJECTS/ARES-DASHBOARD", "box": "ARES",
                    "kind": "project", "path": "/mnt/nvme/PROMETHEUS/PROJECTS/ARES-DASHBOARD", "name": "ARES-DASHBOARD"})
    _chat(st, "MAC", "m1", "/Users/zk/PROMETHEUS/.ARES.a/PROJECTS/ARES-DASHBOARD")
    res = link_workdirs(st)
    assert "ARES:/mnt/nvme/PROMETHEUS/PROJECTS/ARES-DASHBOARD" in _parents(st, "MAC:chat/m1")
    assert st.get_node("MAC:chat/m1")["meta"]["cwd_canon"] == "/mnt/nvme/PROMETHEUS/PROJECTS/ARES-DASHBOARD"
    assert res["linked_existing"] == 1
    st.close()


def test_zeus_sessions_get_a_workdir_chain(tmp_path):
    st = Store(str(tmp_path / "kg.db"))
    u = "/srv/dev-disk-by-uuid-27b8"
    _chat(st, "ZEUS", "z1", u + "/BUSINESS/FAI/bdr", "2026-08-01 · bdr sender fix")
    _chat(st, "ZEUS", "z2", "/srv/mergerfs/PROMETHEUS/BUSINESS/FAI/bdr", "2026-08-05 · bdr replies")
    res = link_workdirs(st)
    wd = "ZEUS:/srv/mergerfs/PROMETHEUS/BUSINESS/FAI/bdr"
    assert wd in _parents(st, "ZEUS:chat/z1") and wd in _parents(st, "ZEUS:chat/z2")
    n = st.get_node(wd)
    assert n["kind"] == "folder" and "2 Claude Code sessions" in n["understanding"]
    assert "bdr replies" in n["understanding"]
    assert "ZEUS:/srv/mergerfs/PROMETHEUS/BUSINESS/FAI" in _parents(st, wd)      # parent chain
    assert res["workdirs_created"] >= 2
    st.close()


def test_idempotent_and_leaves_linked_chats_alone(tmp_path):
    st = Store(str(tmp_path / "kg.db"))
    st.upsert_node({"id": "ARES:/p", "box": "ARES", "kind": "folder", "path": "/p", "name": "p"})
    _chat(st, "ARES", "a1", "/p")
    st.add_edge("ARES:/p", "ARES:chat/a1", "contains")
    _chat(st, "ZEUS", "z1", "/srv/mergerfs/PROMETHEUS/X")
    link_workdirs(st)
    edges1 = st.db.execute("SELECT count(*) FROM edges").fetchone()[0]
    res = link_workdirs(st)
    assert st.db.execute("SELECT count(*) FROM edges").fetchone()[0] == edges1
    assert res["already_linked"] >= 1
    st.close()
