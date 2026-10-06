import json

import pytest

from atlas import prompt_hook as H
from atlas.store import Store


def test_query_text_drops_slash_command_and_pasted_tags():
    assert H.query_text("/zain fix the vpn login on the app") == "fix the vpn login on the app"
    assert H.query_text("<pasted_content id=\"x\">\nphone vpn drops\n</pasted_content id=\"x\">") == "phone vpn drops"
    assert len(H.query_text("word " * 500)) <= H.QUERY_MAX


def test_content_tokens_skip_stopwords_and_short_words():
    assert H.content_tokens("why is the vault swipe broken on my phone?") == ["vault", "swipe", "broken", "phone"]


def test_worth_searching_needs_real_content():
    assert not H.worth_searching("continue")
    assert not H.worth_searching("yes do it")
    assert H.worth_searching("photos videos dont play on the app")


def test_relevant_needs_two_shared_tokens_and_skips_own_session():
    hit = {"id": "ARES:chat/abc", "name": "Fix vault swipe", "understanding": "The phone vault lightbox swipe..."}
    toks = ["vault", "swipe", "phone"]
    assert H.relevant(hit, toks, session_id="zzz")
    assert not H.relevant(hit, toks, session_id="abc")               # this very session
    assert not H.relevant({"id": "x", "name": "Ocean's 11 remake", "understanding": "film"}, ["ocean", "poem"], "s")


def test_first_prompt_only(tmp_path):
    assert H.first_prompt(tmp_path, "s1")
    assert not H.first_prompt(tmp_path, "s1")
    assert H.first_prompt(tmp_path, "s2")


@pytest.fixture
def kg(tmp_path):
    s = Store(str(tmp_path / "kg.db"))
    s.upsert_node({"id": "ARES:chat/old1", "box": "ARES", "kind": "chat", "path": "", "name": "2026-09-19 · Vault swipe fix",
                   "understanding": "Fixed the phone vault lightbox swipe that did not load the next photo.",
                   "meta": {"session": "old1"}})
    s.upsert_node({"id": "ARES:chat/me", "box": "ARES", "kind": "chat", "path": "", "name": "today · vault swipe",
                   "understanding": "this session vault swipe phone", "meta": {"session": "me"}})
    return tmp_path / "kg.db"


def test_run_injects_matching_sessions_and_logs(kg, tmp_path):
    state = tmp_path / "state"
    out = H.run({"session_id": "me", "prompt": "the vault swipe on my phone does not load the next photo"},
                db=str(kg), state_dir=state, embed=lambda q: None, jev_key="")
    ctx = out["hookSpecificOutput"]["additionalContext"]
    assert out["hookSpecificOutput"]["hookEventName"] == "UserPromptSubmit"
    assert "ARES:chat/old1" in ctx and "ARES:chat/me" not in ctx
    log = json.loads((state / "log.jsonl").read_text().splitlines()[0])
    assert log["session"] == "me" and log["hits"] == 1
    # second prompt of the same session: silent
    assert H.run({"session_id": "me", "prompt": "the vault swipe again phone photo"},
                 db=str(kg), state_dir=state, embed=lambda q: None, jev_key="") is None


def test_run_silent_when_nothing_relevant(kg, tmp_path):
    assert H.run({"session_id": "n", "prompt": "write me a poem about the ocean waves"},
                 db=str(kg), state_dir=tmp_path / "st", embed=lambda q: None, jev_key="") is None


def test_pick_puts_sessions_first_and_caps_folders():
    hits = [{"id": f"f{i}", "kind": "folder"} for i in range(4)] + [{"id": "c1", "kind": "chat"}, {"id": "c2", "kind": "gpt-chat"}]
    assert [h["id"] for h in H.pick(hits)] == ["c1", "c2", "f0", "f1"]


def test_relevant_scales_with_long_prompts_and_skips_docs():
    toks = ["ares", "app", "vault", "swipe", "offline", "login", "photos", "phone"]
    weak = {"id": "a", "kind": "chat", "name": "ARES app polish", "understanding": "misc"}
    strong = {"id": "b", "kind": "chat", "name": "Vault swipe + offline login", "understanding": "ARES app"}
    doc = dict(strong, kind="file-content")
    assert not H.relevant(weak, toks, "s")
    assert H.relevant(strong, toks, "s")
    assert not H.relevant(doc, toks, "s")


def test_pick_dedupes_one_session_on_two_boxes():
    hits = [{"id": "ARES:chat/x", "kind": "chat"}, {"id": "ZEUS:chat/x", "kind": "chat"}, {"id": "ARES:chat/y", "kind": "chat"}]
    assert [h["id"] for h in H.pick(hits)] == ["ARES:chat/x", "ARES:chat/y"]


def test_short_prompt_needs_all_but_one_word():
    toks = H.content_tokens("add the hook to zeus and the macs too")
    assert toks == ["add", "hook", "zeus", "macs"]
    email = {"id": "ZEUS:chat/e", "kind": "chat", "name": "ZEUS: cold email", "understanding": "email hook for a prospect"}
    install = {"id": "ARES:chat/i", "kind": "chat", "name": "Add kg hook to ZEUS and Macs", "understanding": "install"}
    assert not H.relevant(email, toks, "s")
    assert H.relevant(install, toks, "s")


def test_relevant_matches_whole_words_not_substrings():
    toks = ["add", "hook", "zeus", "macs"]
    email = {"id": "e", "kind": "chat", "name": "ZEUS cold email", "understanding": "address the hook to the prospect"}
    assert not H.relevant(email, toks, "s")                  # "add" must not match inside "address"
    plural = {"id": "p", "kind": "chat", "name": "photos videos", "understanding": "gallery"}
    assert H.relevant(plural, ["photo", "video", "gallery"], "s")   # 4+ letter prefix still counts


def _jev_http(scores):
    def http(url, body, headers, timeout):
        req = json.loads(body)
        assert req["model"] == H.JEV_MODEL and set(req["questions"]) == set(scores)
        return {"answers": {k: {"noul": v} for k, v in scores.items()}}
    return http


def test_jev_filter_drops_low_scores_and_keeps_order():
    hits = [{"id": "a", "name": "A"}, {"id": "b", "name": "B"}, {"id": "c", "name": "C"}]
    kept = H.jev_filter("prompt", hits, key="sk-or-x", http=_jev_http({"c0": 0.9, "c1": 0.1, "c2": 0.4}))
    assert [h["id"] for h in kept] == ["a", "c"]


def test_jev_filter_falls_back_on_error_or_no_key():
    hits = [{"id": "a", "name": "A"}]
    def boom(*a, **k):
        raise TimeoutError
    assert H.jev_filter("p", hits, key="sk-or-x", http=boom) is None
    assert H.jev_filter("p", hits, key=None, http=boom) is None
    assert H.jev_filter("p", [], key="sk-or-x", http=boom) == []


def test_bounded_embed_returns_fast_and_does_not_block_exit():
    import subprocess, sys, textwrap, time
    code = textwrap.dedent("""
        import sys, time
        sys.path.insert(0, ".")
        from atlas.prompt_hook import _bounded_embed
        print(_bounded_embed(lambda q: time.sleep(30), timeout=0.2)("x"))
    """)
    t = time.monotonic()
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=10)
    assert out.stdout.strip() == "None"
    assert time.monotonic() - t < 5     # the 30 s embed must not keep the process alive


def test_bounded_embed_passes_a_quick_result():
    from atlas.prompt_hook import _bounded_embed
    assert _bounded_embed(lambda q: [1.0, 2.0], timeout=1.0)("x") == [1.0, 2.0]
