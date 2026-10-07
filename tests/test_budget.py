import json, os
import pytest
from atlas import budget as B
from atlas import understanding as U
from atlas.store import Store


@pytest.fixture
def state(tmp_path, monkeypatch):
    monkeypatch.setattr(B, "STATE", str(tmp_path / "budget.json"))
    monkeypatch.setattr(B, "ALERT_FILE", str(tmp_path / "atlas-ALERT"))
    monkeypatch.setattr(B, "DAILY_CALLS", 3)
    monkeypatch.setattr(B, "DAILY_USD", 0.01)
    return tmp_path


def test_daily_call_cap(state):
    assert [B.reserve() for _ in range(4)] == [True, True, True, False]
    assert B.status()["calls"] == 3
    assert "daily cap" in open(state / "atlas-ALERT").read()


def test_dollar_cap(state):
    assert B.reserve()
    B.record_cost(0.02)                      # one expensive call blows the $0.01 cap
    assert B.reserve() is False
    assert B.status()["usd"] == pytest.approx(0.02)


def test_new_day_resets(state, monkeypatch):
    B.reserve(); B.reserve(); B.reserve()
    monkeypatch.setattr(B, "_today", lambda: "2099-01-01")
    assert B.reserve() is True and B.status()["calls"] == 1


def test_llm_refuses_past_cap_and_records_cost(state, monkeypatch):
    monkeypatch.setattr(U, "openrouter_key", lambda: "sk-or-x")
    seen = {}
    def fake(key, body):
        seen["body"] = body
        return {"choices": [{"message": {"content": "ok"}}], "usage": {"cost": 0.004}}
    assert U.llm("hi", http=fake) == "ok"
    assert seen["body"]["usage"] == {"include": True}
    assert B.status()["usd"] == pytest.approx(0.004)
    monkeypatch.setattr(B, "DAILY_CALLS", 1)
    with pytest.raises(U.OutOfCredit):        # BudgetExceeded is an OutOfCredit: callers stop cleanly
        U.llm("again", http=fake)


def test_cost_estimated_when_provider_omits_it(state, monkeypatch):
    monkeypatch.setattr(U, "openrouter_key", lambda: "sk-or-x")
    fake = lambda key, body: {"choices": [{"message": {"content": "ok"}}],
                              "usage": {"prompt_tokens": 1_000_000, "completion_tokens": 0}}
    U.llm("hi", http=fake)
    assert B.status()["usd"] == pytest.approx(U.PRICE_IN_PER_M)


def _session(root, sid, text="fix caddy"):
    d = os.path.join(root, "proj"); os.makedirs(d, exist_ok=True)
    p = os.path.join(d, f"{sid}.jsonl")
    with open(p, "w") as f:
        f.write(json.dumps({"type": "user", "cwd": "/x", "message": {"content": text}}) + "\n")
    return p


def test_path_change_alone_never_resummarizes(tmp_path, monkeypatch, state):
    from atlas.chats import index_chats
    calls = []
    monkeypatch.setattr("atlas.understanding.summarize_session",
                        lambda *a, **k: calls.append(1) or "Summary.")
    st = Store(str(tmp_path / "kg.db"))
    a = _session(str(tmp_path / "rootA"), "s1")
    index_chats(st, projects_root=str(tmp_path / "rootA"), box="ARES", min_idle=0, embed_fn=lambda t: None)
    b_root = tmp_path / "rootB" / "proj"; b_root.mkdir(parents=True)
    b = str(b_root / "s1.jsonl")
    os.link(a, b)                                    # same file, same mtime+size, new path
    for root in ("rootB", "rootA", "rootB"):         # the old ping-pong
        index_chats(st, projects_root=str(tmp_path / root), box="ARES", min_idle=0, embed_fn=lambda t: None)
    assert len(calls) == 1
    st.close()


def test_requeued_summaries_alert_and_skip_but_first_summaries_do_not(tmp_path, monkeypatch, state):
    from atlas.chats import index_chats
    import atlas.chats as C
    monkeypatch.setattr(C, "QUEUE_ALERT", 2)
    calls = []
    monkeypatch.setattr("atlas.understanding.summarize_session",
                        lambda *a, **k: calls.append(1) or "Summary.")
    st = Store(str(tmp_path / "kg.db"))
    paths = [_session(str(tmp_path / "r"), f"s{i}") for i in range(3)]
    run = lambda: index_chats(st, projects_root=str(tmp_path / "r"), box="ARES", min_idle=0,
                              embed_fn=lambda t: None)
    res = run()                                       # 3 FIRST summaries: a backlog, not a loop
    assert len(calls) == 3 and res["queue_blocked"] is False
    for p in paths:                                   # all 3 already-summarized sessions change at once
        with open(p, "a") as f:
            f.write(json.dumps({"type": "user", "message": {"content": "more"}}) + "\n")
    res = run()
    assert len(calls) == 3 and res["queue_blocked"] is True and res["requeued"] == 3
    assert "3 already-summarized sessions" in open(state / "atlas-ALERT").read()
    monkeypatch.setenv("KG_ALLOW_BIG_QUEUE", "1")    # deliberate re-summary override
    run()
    assert len(calls) == 6
    st.close()


def test_history_first_summaries_are_not_blocked(tmp_path, monkeypatch, state):
    import atlas.chats as C
    from atlas.history import index_history
    monkeypatch.setattr(C, "QUEUE_ALERT", 0)          # first summaries never trip it
    calls = []
    monkeypatch.setattr("atlas.understanding.summarize_session",
                        lambda *a, **k: calls.append(1) or "S.")
    h = tmp_path / "history.jsonl"
    h.write_text("\n".join(json.dumps({"display": f"ask {i}", "project": "/x", "sessionId": f"s{i}",
                                       "timestamp": 1_700_000_000_000 + i}) for i in range(3)) + "\n")
    st = Store(str(tmp_path / "kg.db"))
    res = index_history(st, str(h), box="ARES", min_idle=0, embed_fn=lambda t: None)
    assert len(calls) == 3 and res["added"] == 3
    st.close()
