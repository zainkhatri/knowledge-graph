from atlas.store import Store
from atlas.names import is_noise_prompt, pick_title, fix_names


def test_noise_prompt_rules():
    for x in ["/resume", "/model sonnet", "/logout", "  /mcp  ",
              "Base directory for this skill: /root/.claude/skills/x  # X"]:
        assert is_noise_prompt(x), x
    for x in ["/frontend-design mobile optimize the entire page",
              "/Users/zainkhatri/Desktop/ZAIN take WORK folder", "fix the caddy config", ""]:
        assert not is_noise_prompt(x) or x == "", x


def test_pick_title_skips_commands_then_falls_back_to_summary():
    assert pick_title(["/resume", "/model opus", "why is zeus down"]) == "why is zeus down"
    assert pick_title(["/resume"], summary="ZEUS was asleep. Woke it via WoL.") == "ZEUS was asleep."
    assert pick_title(["/resume"]) == "/resume"          # nothing better available


def test_fix_names_renames_only_bad_ones(tmp_path):
    st = Store(str(tmp_path / "kg.db"))
    def chat(nid, name, asks, und="Fixed the photo grid lag. More text.", status="live"):
        st.upsert_node({"id": nid, "box": "ARES", "kind": "chat", "path": nid, "name": name,
                        "understanding": und, "status": status, "meta": {"asks": asks}})
    chat("a", "2026-05-01 · /resume", ["/resume", "photo grid is laggy on iphone"])
    chat("b", "2026-05-02 · /resume", ["/resume"])
    chat("c", "2026-05-03 · Fix Caddy", ["/resume", "x"])            # already a real title
    res = fix_names(st)
    assert st.get_node("a")["name"] == "2026-05-01 · photo grid is laggy on iphone"
    assert st.get_node("b")["name"] == "2026-05-02 · Fixed the photo grid lag."
    assert st.get_node("c")["name"] == "2026-05-03 · Fix Caddy"
    assert res["renamed"] == 2
    assert [h["id"] for h in st.search("laggy iphone", embed_fn=lambda t, http=None: None)] == ["a"]
    st.close()
