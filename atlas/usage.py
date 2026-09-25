"""Is the graph earning its keep? Weekly usage stats from the session archive.

Scans archived Claude Code transcripts (PERSONAL/CLAUDE-CODE-SESSIONS) modified in the
last N days and counts: sessions that called homelab-kg, graph calls, empty graph
results, tokens spent on graph results, and filesystem-discovery calls (Read/Grep/Glob
and grep/find/ls/cat in Bash) — the work the graph is supposed to replace.
Stdlib-only.
"""
import glob, json, os, time

KG_PREFIX = "mcp__homelab-kg__"
DISCOVERY_TOOLS = ("Read", "Grep", "Glob")
DISCOVERY_CMDS = ("grep ", "rg ", "find ", "ls ", "cat ", "fd ")
EMPTY_MAX = 80                     # a result this short is "[]" / "null" / an error
MAX_LINES = 3_000_000


def _result_text(content):
    return content if isinstance(content, str) else json.dumps(content)


def scan_session(path):
    s = {"kg_calls": 0, "kg_empty": 0, "kg_result_chars": 0, "discovery_calls": 0}
    names = {}
    try:
        f = open(path, errors="ignore")
    except OSError:
        return s
    with f:
        for n, line in enumerate(f):
            if n >= MAX_LINES:
                break
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
    return s


def report(archive_root, days=7):
    cut = time.time() - days * 86400
    files = [p for p in glob.glob(os.path.join(archive_root, "*", "*", "*.jsonl"))
             if os.path.getmtime(p) >= cut]
    tot = {"kg_calls": 0, "kg_empty": 0, "kg_result_chars": 0, "discovery_calls": 0}
    using = 0
    for p in files:
        s = scan_session(p)
        using += s["kg_calls"] > 0
        for k in tot:
            tot[k] += s[k]
    n = len(files)
    return {
        "date": time.strftime("%Y-%m-%d"), "days": days, "sessions": n,
        "sessions_using_graph": using,
        "graph_use_pct": round(100.0 * using / n, 1) if n else 0.0,
        "kg_calls": tot["kg_calls"],
        "kg_empty_pct": round(100.0 * tot["kg_empty"] / tot["kg_calls"], 1) if tot["kg_calls"] else 0.0,
        "kg_result_tokens": tot["kg_result_chars"] // 4,
        "discovery_calls": tot["discovery_calls"],
    }
