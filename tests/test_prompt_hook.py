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
                db=str(kg), state_dir=state, embed=lambda q: None)
    ctx = out["hookSpecificOutput"]["additionalContext"]
    assert out["hookSpecificOutput"]["hookEventName"] == "UserPromptSubmit"
    assert "ARES:chat/old1" in ctx and "ARES:chat/me" not in ctx
    log = json.loads((state / "log.jsonl").read_text().splitlines()[0])
    assert log["session"] == "me" and log["hits"] == 1
    # second prompt of the same session: silent
    assert H.run({"session_id": "me", "prompt": "the vault swipe again phone photo"},
                 db=str(kg), state_dir=state, embed=lambda q: None) is None


def test_run_silent_when_nothing_relevant(kg, tmp_path):
    assert H.run({"session_id": "n", "prompt": "write me a poem about the ocean waves"},
                 db=str(kg), state_dir=tmp_path / "st", embed=lambda q: None) is None


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
