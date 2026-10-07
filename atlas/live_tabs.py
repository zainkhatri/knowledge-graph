"""Live awareness between concurrent Claude Code sessions ("tabs").

The graph learns about a session only after the next sync; tabs that are already open
never hear about each other. On 2026-10-07 that caused real collisions: an in-place edit
of a script another job was running, unexpected uncommitted edits in shared files, and
another session's commits riding along on a push. This reads the transcripts that are
being written RIGHT NOW (mtime within ACTIVE_SECS) and:
  - run_prompt  (UserPromptSubmit): lists other active sessions with the files they
    edited, but only when that picture changed since this session last saw it;
  - run_pretool (PreToolUse on Edit/Write): asks the user before editing a file another
    active session edited.
Local transcripts are live; other boxes come from the session archive (synced every
10 min). Reads only the tail of each transcript. Stdlib-only.
"""
import glob
import hashlib
import json
import os
import time

ACTIVE_SECS = 30 * 60
TAIL_BYTES = 400_000
MAX_SESSIONS = 6
MAX_FILES = 8
EDIT_TOOLS = frozenset({"Edit", "Write", "MultiEdit", "NotebookEdit"})
ARCHIVE = "/mnt/nvme/PROMETHEUS/PERSONAL/CLAUDE-CODE-SESSIONS"


def default_roots():
    roots = [("ARES", "/root/.claude/projects")]
    try:                                                       # LXC 101 (ARES app chats), live
        import subprocess
        pid = subprocess.run(["lxc-info", "-n", "101", "-p", "-H"], capture_output=True,
                             text=True, timeout=2).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pid = ""
    p = f"/proc/{pid}/root/root/.claude/projects"
    if pid.isdigit() and os.path.isdir(p):
        roots.append(("ARES", p))
    for src in ("ZEUS-zain", "ZEUS-root", "EROS", "MAC-air", "MAC-prometheon"):
        d = os.path.join(ARCHIVE, src)
        if os.path.isdir(d):
            roots.append((src.split("-")[0], d))
    return roots


def _text(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(c.get("text", "") for c in content if isinstance(c, dict) and c.get("type") == "text")
    return ""


def scan_session(path):
    """Title, cwd, last human prompt, entrypoint and edited files from the transcript tail."""
    out = {"title": "", "cwd": "", "last_prompt": "", "entry": "", "files": set()}
    try:
        size = os.path.getsize(path)
        with open(path, "rb") as f:
            f.seek(max(0, size - TAIL_BYTES))
            raw = f.read().decode("utf-8", "ignore")
    except OSError:
        return out
    lines = raw.split("\n")
    if size > TAIL_BYTES:
        lines = lines[1:]                                       # first line is partial
    for line in lines:
        if not line.strip():
            continue
        try:
            d = json.loads(line)
        except ValueError:
            continue
        out["entry"] = d.get("entrypoint") or out["entry"]
        out["cwd"] = d.get("cwd") or out["cwd"]
        t = d.get("type")
        if t in ("ai-title", "custom-title"):
            out["title"] = d.get("customTitle") or d.get("aiTitle") or out["title"]
        elif t == "user":
            txt = _text((d.get("message") or {}).get("content")).strip()
            if txt and not txt.startswith("<") and "tool_result" not in txt[:40]:
                out["last_prompt"] = " ".join(txt.split())[:120]
        elif t == "assistant":
            for b in (d.get("message") or {}).get("content") or []:
                if isinstance(b, dict) and b.get("type") == "tool_use" and b.get("name") in EDIT_TOOLS:
                    fp = (b.get("input") or {}).get("file_path") or (b.get("input") or {}).get("notebook_path")
                    if fp:
                        out["files"].add(fp)
    return out


def active_sessions(roots, now=None, exclude_sid=""):
    now = now if now is not None else time.time()
    found = []
    for box, root in roots:
        for p in glob.glob(os.path.join(root, "*", "*.jsonl")):
            try:
                mt = os.path.getmtime(p)
            except OSError:
                continue
            sid = os.path.splitext(os.path.basename(p))[0]
            if now - mt > ACTIVE_SECS or sid == exclude_sid:
                continue
            found.append((mt, box, sid, p))
    found.sort(reverse=True)
    seen, uniq = set(), []
    for row in found:                # alias (symlinked) project dirs show a session twice
        if row[2] not in seen:
            seen.add(row[2]); uniq.append(row)
    out = []
    for mt, box, sid, p in uniq[:MAX_SESSIONS]:
        s = scan_session(p)
        out.append({"sid": sid, "box": box, "mins": int((now - mt) // 60),
                    "title": s["title"] or s["last_prompt"], "cwd": s["cwd"],
                    "headless": s["entry"].startswith("sdk"), "files": sorted(s["files"])})
    return out


def render(sessions):
    lines = ["Other Claude sessions active in the last 30 min (live-tabs). Do not edit their "
             "files without asking; fetch/check their commits before pushing:"]
    for s in sessions:
        tag = " · headless" if s["headless"] else ""
        lines.append(f"- {s['box']} · {s['mins']} min ago{tag} · {s['title'][:90]} · cwd {s['cwd']}")
        if s["files"]:
            shown = ", ".join(s["files"][:MAX_FILES])
            more = f" (+{len(s['files']) - MAX_FILES})" if len(s["files"]) > MAX_FILES else ""
            lines.append(f"  edited: {shown}{more}")
    return "\n".join(lines)


def run_prompt(inp, state_dir, roots=None, now=None):
    sid = inp.get("session_id") or ""
    sessions = active_sessions(roots or default_roots(), now, exclude_sid=sid)
    sig = hashlib.sha256(json.dumps([(s["sid"], s["files"], s["title"]) for s in sessions])
                         .encode()).hexdigest()
    os.makedirs(state_dir, exist_ok=True)
    state = os.path.join(state_dir, (sid or "unknown") + ".sig")
    try:
        last = open(state).read()
    except OSError:
        last = None
    with open(state, "w") as f:
        f.write(sig)
    if not sessions or sig == last:
        return None
    return {"hookSpecificOutput": {"hookEventName": "UserPromptSubmit",
                                   "additionalContext": render(sessions)}}


def run_pretool(inp, roots=None, now=None):
    if inp.get("tool_name") not in EDIT_TOOLS:
        return None
    ti = inp.get("tool_input") or {}
    target = ti.get("file_path") or ti.get("notebook_path")
    if not target:
        return None
    for s in active_sessions(roots or default_roots(), now, exclude_sid=inp.get("session_id") or ""):
        if target in s["files"]:
            reason = (f"Another Claude session edited this file {s['mins']} min ago "
                      f"({s['box']}: {s['title'][:80]}). Editing it now may collide with "
                      f"that work — confirm to proceed.")
            return {"hookSpecificOutput": {"hookEventName": "PreToolUse",
                                           "permissionDecision": "ask",
                                           "permissionDecisionReason": reason}}
    return None
