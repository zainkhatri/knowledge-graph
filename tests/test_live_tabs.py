import json
import os
import time

from atlas import live_tabs as L

NOW = 2_000_000_000.0


def _session(root, proj, sid, lines, age_s, entry="cli"):
    d = os.path.join(root, proj); os.makedirs(d, exist_ok=True)
    p = os.path.join(d, f"{sid}.jsonl")
    with open(p, "w") as f:
        for l in lines:
            l.setdefault("entrypoint", entry)
            f.write(json.dumps(l) + "\n")
    os.utime(p, (NOW - age_s, NOW - age_s))
    return p


def _user(t, cwd="/mnt/x/ARES-DASHBOARD"):
    return {"type": "user", "cwd": cwd, "message": {"content": t}}


def _edit(path, tool="Edit"):
    return {"type": "assistant", "message": {"content": [
        {"type": "tool_use", "name": tool, "input": {"file_path": path}}]}}


def _title(t):
    return {"type": "ai-title", "aiTitle": t}


def _roots(tmp_path):
    return [("ARES", str(tmp_path / "projects"))]


def test_lists_other_active_sessions_with_files(tmp_path):
    r = str(tmp_path / "projects")
    _session(r, "p", "me", [_user("my task")], 10)
    _session(r, "p", "other", [_title("Fix photo vision"), _user("speed up photos"),
                               _edit("/repo/system/kg-photo-vision.sh"), _edit("/repo/app.py", "Write")], 120)
    _session(r, "p", "stale", [_user("old")], 3 * 3600)
    s = L.active_sessions(_roots(tmp_path), NOW, exclude_sid="me")
    assert [x["sid"] for x in s] == ["other"]
    assert s[0]["title"] == "Fix photo vision"
    assert s[0]["files"] == ["/repo/app.py", "/repo/system/kg-photo-vision.sh"]
    assert s[0]["mins"] == 2


def test_prompt_hook_only_speaks_when_something_changed(tmp_path):
    r = str(tmp_path / "projects")
    _session(r, "p", "other", [_title("Vault work"), _edit("/repo/app.py")], 60)
    inp = {"session_id": "me", "cwd": "/repo", "prompt": "hi"}
    out = L.run_prompt(inp, str(tmp_path / "state"), _roots(tmp_path), NOW)
    ctx = out["hookSpecificOutput"]["additionalContext"]
    assert "Vault work" in ctx and "app.py" in ctx
    assert L.run_prompt(inp, str(tmp_path / "state"), _roots(tmp_path), NOW) is None   # unchanged
    _session(r, "p", "other", [_title("Vault work"), _edit("/repo/app.py"), _edit("/repo/new.py")], 30)
    assert "new.py" in L.run_prompt(inp, str(tmp_path / "state"), _roots(tmp_path), NOW)[
        "hookSpecificOutput"]["additionalContext"]


def test_prompt_hook_silent_when_alone(tmp_path):
    r = str(tmp_path / "projects")
    _session(r, "p", "me", [_user("x")], 5)
    assert L.run_prompt({"session_id": "me"}, str(tmp_path / "s"), _roots(tmp_path), NOW) is None


def test_pretool_asks_before_editing_a_file_another_session_touched(tmp_path):
    r = str(tmp_path / "projects")
    _session(r, "p", "other", [_title("Photo windows"), _edit("/repo/system/kg-photo-vision.sh")], 90)
    inp = {"session_id": "me", "tool_name": "Edit",
           "tool_input": {"file_path": "/repo/system/kg-photo-vision.sh"}}
    out = L.run_pretool(inp, _roots(tmp_path), NOW)["hookSpecificOutput"]
    assert out["permissionDecision"] == "ask" and "Photo windows" in out["permissionDecisionReason"]
    inp["tool_input"]["file_path"] = "/repo/untouched.py"
    assert L.run_pretool(inp, _roots(tmp_path), NOW) is None
    inp["session_id"] = "other"                                    # its own edits never warn
    inp["tool_input"]["file_path"] = "/repo/system/kg-photo-vision.sh"
    assert L.run_pretool(inp, _roots(tmp_path), NOW) is None


def test_reads_only_the_tail_of_huge_transcripts(tmp_path):
    r = str(tmp_path / "projects")
    filler = [_user("x" * 2000) for _ in range(400)]                # ~800 KB before the edit
    _session(r, "p", "big", filler + [_edit("/repo/late.py")], 60)
    s = L.active_sessions(_roots(tmp_path), NOW, exclude_sid="me")
    assert s[0]["files"] == ["/repo/late.py"]


def test_session_reached_via_alias_dir_is_listed_once(tmp_path):
    r = tmp_path / "projects"
    _session(str(r), "real", "other", [_title("One tab"), _edit("/repo/a.py")], 60)
    os.symlink(str(r / "real"), str(r / "alias"))                 # ARES-FAILOVER-style alias
    s = L.active_sessions(_roots(tmp_path), NOW, exclude_sid="me")
    assert [x["sid"] for x in s] == ["other"]
