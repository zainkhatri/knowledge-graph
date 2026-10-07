import json, os, time
from atlas.usage import scan_session, report


def _write(tmp_path, name, blocks):
    d = tmp_path / "ARES" / "proj"; d.mkdir(parents=True, exist_ok=True)
    p = d / f"{name}.jsonl"
    with open(p, "w") as f:
        for role, content in blocks:
            f.write(json.dumps({"type": role, "message": {"content": content}}) + "\n")
    return str(p)


def _use(i, name, inp):
    return ("assistant", [{"type": "tool_use", "id": f"t{i}", "name": name, "input": inp}])


def _res(i, text):
    return ("user", [{"type": "tool_result", "tool_use_id": f"t{i}", "content": text}])


def test_scan_counts_graph_and_discovery(tmp_path):
    p = _write(tmp_path, "s1", [
        _use(1, "mcp__homelab-kg__kg_search", {"query": "caddy"}), _res(1, "[]"),
        _use(2, "mcp__homelab-kg__kg_search", {"query": "tls"}), _res(2, "x" * 400),
        _use(3, "Grep", {"pattern": "caddy"}), _res(3, "a"),
        _use(4, "Bash", {"command": "find / -name Caddyfile"}), _res(4, "b"),
        _use(5, "Bash", {"command": "systemctl restart caddy"}), _res(5, "c"),
    ])
    s = scan_session(p)
    assert s["kg_calls"] == 2 and s["kg_empty"] == 1
    assert s["kg_result_chars"] == 2 + 400
    assert s["discovery_calls"] == 2


def test_report_aggregates_recent_only(tmp_path):
    _write(tmp_path, "a", [_use(1, "mcp__homelab-kg__kg_search", {"query": "q"}), _res(1, "hit" * 50)])
    _write(tmp_path, "b", [_use(1, "Read", {"file_path": "/x"}), _res(1, "y")])
    old = _write(tmp_path, "c", [_use(1, "Read", {"file_path": "/x"}), _res(1, "y")])
    t = time.time() - 30 * 86400
    os.utime(old, (t, t))
    r = report(str(tmp_path), days=7)
    assert r["sessions"] == 2 and r["sessions_using_graph"] == 1
    assert r["graph_use_pct"] == 50.0 and r["kg_empty_pct"] == 0.0
    assert r["discovery_calls"] == 1


def _write_raw(tmp_path, name, lines, mtime=None):
    d = tmp_path / "ARES" / "proj"; d.mkdir(parents=True, exist_ok=True)
    p = d / f"{name}.jsonl"
    p.write_text("\n".join(json.dumps(l) for l in lines) + "\n")
    return str(p)


def _ctx(text):
    return {"type": "attachment", "entrypoint": "cli",
            "attachment": {"type": "hook_additional_context", "content": [text]}}


def _tool(i, name, entry="cli"):
    return {"type": "assistant", "entrypoint": entry,
            "message": {"content": [{"type": "tool_use", "id": f"t{i}", "name": name, "input": {}}]}}


def test_scan_detects_headless_and_auto_context(tmp_path):
    p = _write_raw(tmp_path, "h", [_tool(1, "Read", entry="sdk-cli")])
    s = scan_session(p)
    assert s["headless"] is True and s["auto_ctx"] == 0
    p = _write_raw(tmp_path, "i", [_ctx("ATLAS was searched automatically ... (homelab-kg). Past work"),
                                   _ctx("unrelated hook text"), _tool(1, "Grep")])
    s = scan_session(p)
    assert s["headless"] is False and s["auto_ctx"] == 1


def test_report_splits_interactive_and_headless(tmp_path):
    _write_raw(tmp_path, "i1", [_ctx("Recent past Claude Code sessions for this project (homelab-kg)."),
                                _tool(1, "Grep")])                         # auto context only
    _write_raw(tmp_path, "i2", [_tool(1, "mcp__homelab-kg__kg_search")])   # explicit call
    _write_raw(tmp_path, "i3", [_tool(1, "Read")])                         # no graph
    _write_raw(tmp_path, "h1", [_tool(1, "Read", entry="sdk-cli")])        # headless
    r = report(str(tmp_path), days=7)
    assert r["interactive"]["sessions"] == 3 and r["headless"]["sessions"] == 1
    assert r["interactive"]["explicit_use_pct"] == 33.3       # i2
    assert r["interactive"]["any_use_pct"] == 66.7            # i1 + i2
    assert r["interactive"]["discovery_calls"] == 2
    assert r["headless"]["any_use_pct"] == 0.0
    assert r["sessions"] == 4                                 # totals kept for the old series
