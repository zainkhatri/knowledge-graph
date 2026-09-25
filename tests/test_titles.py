from atlas.store import Store
from atlas.titles import gen_titles, clean_title


def _chat(st, nid, name, und="Fixed ARES photo grid lag on iPhone.", title="", status="live", cwd="/p"):
    st.upsert_node({"id": nid, "box": "ARES", "kind": "chat", "path": nid, "name": name,
                    "understanding": und, "status": status,
                    "meta": {"title": title, "asks": ["photo grid is laggy", "use ARES_PASSWORD=hunter2x"],
                             "cwd": cwd}})


def test_clean_title():
    assert clean_title('  "Fix ARES photo grid lag."  ') == "Fix ARES photo grid lag"
    assert clean_title("Title: Moonlight audio fix") == "Moonlight audio fix"
    assert clean_title("") is None
    assert len(clean_title("word " * 40)) <= 80


def test_titles_only_untitled_live_non_automated(tmp_path, monkeypatch):
    st = Store(str(tmp_path / "kg.db"))
    _chat(st, "a", "2026-05-01 · what can i do with ruflo")
    _chat(st, "b", "2026-05-02 · Has title", title="Real AI title")
    _chat(st, "c", "2026-05-03 · raw one", status="raw")
    _chat(st, "d", "2026-05-04 · classify", cwd="/home/zain/.config/cc-proxy/ccwd")
    prompts = []
    def fake_llm(prompt, max_tokens=400, http=None):
        prompts.append(prompt); return '"ARES photo grid lag fix."'
    monkeypatch.setattr("atlas.understanding.llm", fake_llm)
    res = gen_titles(st, workers=2)
    assert res["titled"] == 1 and len(prompts) == 1
    n = st.get_node("a")
    assert n["name"] == "2026-05-01 · ARES photo grid lag fix"
    assert n["meta"]["gen_title"] == "ARES photo grid lag fix"
    assert "Fixed ARES photo grid lag" in prompts[0]
    assert st.get_node("b")["name"] == "2026-05-02 · Has title"
    assert [h["id"] for h in st.search("grid lag fix", embed_fn=lambda t, http=None: None)][0] == "a"
    # idempotent: already titled nodes are skipped next run
    assert gen_titles(st, workers=2)["titled"] == 0
    st.close()


def test_gen_title_survives_reindex(tmp_path, monkeypatch):
    import json, os
    from atlas.chats import index_chats
    root = tmp_path / "p" / "proj"; root.mkdir(parents=True)
    f = root / "s1.jsonl"
    f.write_text(json.dumps({"type": "user", "cwd": "/x", "timestamp": "2026-05-01T00:00:00Z",
                             "message": {"content": "what can i do with ruflo"}}) + "\n")
    st = Store(str(tmp_path / "kg.db"))
    index_chats(st, projects_root=str(tmp_path / "p"), box="ARES", summarize=False)
    with st.db:
        st.db.execute("UPDATE nodes SET status='live', understanding='Photo grid work.' WHERE id='ARES:chat/s1'")
    monkeypatch.setattr("atlas.understanding.llm", lambda p, max_tokens=400, http=None: "ARES photo grid lag fix")
    gen_titles(st, workers=1)
    with open(f, "a") as fh:                                         # session continues → re-index
        fh.write(json.dumps({"type": "user", "message": {"content": "more work"}}) + "\n")
    index_chats(st, projects_root=str(tmp_path / "p"), box="ARES", summarize=False)
    assert st.get_node("ARES:chat/s1")["name"].endswith("ARES photo grid lag fix")
    st.close()
