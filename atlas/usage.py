"""Is the graph earning its keep? Weekly usage stats from the session archive.

Scans archived Claude Code transcripts (PERSONAL/CLAUDE-CODE-SESSIONS) modified in the
last N days and counts: sessions that called homelab-kg, graph calls, empty graph
results, tokens spent on graph results, and filesystem-discovery calls (Read/Grep/Glob
and grep/find/ls/cat in Bash) — the work the graph is supposed to replace.
Stdlib-only.
"""
import glob, json, os, re, time

KG_PREFIX = "mcp__homelab-kg__"
DISCOVERY_TOOLS = ("Read", "Grep", "Glob")
DISCOVERY_CMDS = ("grep ", "rg ", "find ", "ls ", "cat ", "fd ")
EMPTY_MAX = 80                     # a result this short is "[]" / "null" / an error
AUTO_MARK = "homelab-kg"           # both ATLAS hooks (first-prompt search, SessionStart list) say this
_ENTRY = re.compile(r'"entrypoint":\s*"([a-z-]+)"')   # cli = interactive, sdk-cli = headless claude -p
MAX_LINES = 3_000_000


def _result_text(content):
    return content if isinstance(content, str) else json.dumps(content)


def scan_session(path):
    s = {"kg_calls": 0, "kg_empty": 0, "kg_result_chars": 0, "discovery_calls": 0,
         "headless": False, "auto_ctx": 0}
    entry = None
    names = {}
    try:
        f = open(path, errors="ignore")
    except OSError:
        return s
    with f:
        for n, line in enumerate(f):
            if n >= MAX_LINES:
                break
            if entry is None:
                m = _ENTRY.search(line)
                if m:
                    entry = m.group(1)
            if '"hook_additional_context"' in line and AUTO_MARK in line:
                s["auto_ctx"] += 1                 # graph context injected by a hook (no tool call)
                continue
            if '"tool_use"' not in line and '"tool_result"' not in line:
                continue
            try:
                content = json.loads(line).get("message", {}).get("content")
            except Exception:
                continue
            if not isinstance(content, list):
                continue
            for b in content:
                if not isinstance(b, dict):
                    continue
                if b.get("type") == "tool_use":
                    name = b.get("name", ""); names[b.get("id")] = name
                    if name.startswith(KG_PREFIX):
                        s["kg_calls"] += 1
                    elif name in DISCOVERY_TOOLS or (name == "Bash" and any(
                            c in str((b.get("input") or {}).get("command", "")) for c in DISCOVERY_CMDS)):
                        s["discovery_calls"] += 1
                elif b.get("type") == "tool_result" and \
                        names.get(b.get("tool_use_id"), "").startswith(KG_PREFIX):
                    size = len(_result_text(b.get("content")))
                    s["kg_result_chars"] += size
                    s["kg_empty"] += size < EMPTY_MAX
    s["headless"] = (entry or "").startswith("sdk")
    return s


def _group(stats):
    n = len(stats)
    calls = sum(x["kg_calls"] for x in stats)
    explicit = sum(1 for x in stats if x["kg_calls"])
    anyuse = sum(1 for x in stats if x["kg_calls"] or x["auto_ctx"])
    pct = lambda k: round(100.0 * k / n, 1) if n else 0.0
    return {"sessions": n, "explicit_use_pct": pct(explicit), "any_use_pct": pct(anyuse),
            "auto_ctx_sessions": sum(1 for x in stats if x["auto_ctx"]),
            "kg_calls": calls,
            "kg_empty_pct": round(100.0 * sum(x["kg_empty"] for x in stats) / calls, 1) if calls else 0.0,
            "kg_result_tokens": sum(x["kg_result_chars"] for x in stats) // 4,
            "discovery_calls": sum(x["discovery_calls"] for x in stats)}


def report(archive_root, days=7):
    """Weekly usage. `interactive` vs `headless` (claude -p: autofix councils, scripts) are
    split, and `any_use_pct` counts hook-injected graph context as use. The flat top-level
    keys are the original series (all sessions, explicit calls only) for comparison."""
    cut = time.time() - days * 86400
    files = [p for p in glob.glob(os.path.join(archive_root, "*", "*", "*.jsonl"))
             if os.path.getmtime(p) >= cut]
    stats = [scan_session(p) for p in files]
    allg = _group(stats)
    return {
        "date": time.strftime("%Y-%m-%d"), "days": days, "sessions": allg["sessions"],
        "sessions_using_graph": sum(1 for x in stats if x["kg_calls"]),
        "graph_use_pct": allg["explicit_use_pct"],
        "kg_calls": allg["kg_calls"], "kg_empty_pct": allg["kg_empty_pct"],
        "kg_result_tokens": allg["kg_result_tokens"], "discovery_calls": allg["discovery_calls"],
        "interactive": _group([x for x in stats if not x["headless"]]),
        "headless": _group([x for x in stats if x["headless"]]),
    }
