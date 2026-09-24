import json
from atlas.transcript import read, digest


def _write(tmp_path, entries):
    p = tmp_path / "s.jsonl"
    p.write_text("\n".join(json.dumps(e) for e in entries) + "\n")
    return str(p)


def _user(txt, ts="2026-09-01T00:00:00Z"):
    return {"type": "user", "cwd": "/proj", "timestamp": ts, "message": {"content": txt}}


def _asst(txt):
    return {"type": "assistant", "message": {"content": [{"type": "text", "text": txt}]}}


def test_reads_whole_file_title_and_filters_noise(tmp_path):
    entries = [_user("first ask")]
    for i in range(600):                               # well past the old 400-line head
        entries.append(_asst(f"reply {i}"))
    entries += [
        _user("<command-name>/clear</command-name>"),
        _user([{"type": "tool_result", "content": "x"}]),
        _user("late ask about caddy", ts="2026-09-02T00:00:00Z"),
        {"type": "ai-title", "aiTitle": "Fix Caddy"},
    ]
    tr = read(_write(tmp_path, entries))
    assert tr["asks"] == ["first ask", "late ask about caddy"]
    assert tr["title"] == "Fix Caddy"
    assert tr["cwd"] == "/proj"
    assert tr["first_ts"].startswith("2026-09-01") and tr["last_ts"].startswith("2026-09-02")
    assert ("CLAUDE", "reply 599") in tr["turns"]


def test_custom_title_wins(tmp_path):
    tr = read(_write(tmp_path, [{"type": "ai-title", "aiTitle": "a"},
                                {"type": "custom-title", "customTitle": "b"}]))
    assert tr["title"] == "b"


def test_missing_file_is_empty():
    tr = read("/nonexistent/x.jsonl")
    assert tr["asks"] == [] and tr["turns"] == []


def test_digest_fits_budget_keeps_head_and_tail():
    turns = [("USER", f"ask {i} " + "x" * 200) for i in range(500)]
    d = digest(turns, max_chars=6000)
    assert len(d) <= 6500
    assert d.startswith("USER: ask 0 ")
    assert "ask 499 " in d
    assert "sampled" in d


def test_digest_short_is_verbatim():
    turns = [("USER", "hi"), ("CLAUDE", "hello")]
    assert digest(turns) == "USER: hi\nCLAUDE: hello"
