#!/usr/bin/env python3
"""Minimal stdio JSON-RPC 2.0 MCP server for homelab-kg.
Handles: initialize, tools/list, tools/call.
No external deps beyond stdlib + atlas.store.
"""
import json, os, sys, traceback

ROOT = os.path.dirname(__file__)
DB_PATH = os.environ.get("KG_DB", os.path.join(ROOT, "data", "homelab_kg.db"))

sys.path.insert(0, ROOT)
from atlas.store import Store

_store = None

def store():
    global _store
    if _store is None:
        _store = Store(DB_PATH)
    return _store


# ── tool implementations ──────────────────────────────────────────────────────

SEARCH_LIMIT = 8        # was 15: most callers read the top few hits
SEARCH_CLIP = 300       # chars of understanding per hit; kg_get returns the full text
ASK_CLIP = 300
_GET_DROP = ("embedding", "fingerprint", "size", "status")   # binary/bookkeeping; ~15K chars of noise


def _clip(text, n):
    text = (text or "").strip()
    return text if len(text) <= n else text[:n].rstrip() + "…"


def kg_search(query: str, limit: int = SEARCH_LIMIT) -> list:
    rows = store().search(query, limit=int(limit))
    out = []
    for r in rows:
        hit = {"id": r["id"], "kind": r.get("kind"), "name": r.get("name"),
               "understanding": _clip(r.get("understanding"), SEARCH_CLIP)}
        if r.get("kind") not in ("chat", "gpt-chat", "claude-chat") and r.get("path"):
            hit["path"] = r["path"]              # folders/files: the path IS the answer
        out.append(hit)
    return out


def kg_get(id: str) -> dict | None:
    node = store().get_node(id)
    if node is None:
        return None
    node = {k: v for k, v in node.items() if k not in _GET_DROP}
    meta = node.get("meta")
    if isinstance(meta, dict) and isinstance(meta.get("asks"), list):
        node["meta"] = dict(meta, asks=[_clip(a, ASK_CLIP) for a in meta["asks"]])
    return node


def kg_neighbors(id: str, type: str | None = None) -> list:
    return store().neighbors(id, etype=type or None)


def kg_tree(id: str, depth: int = 2) -> dict | None:
    # ponytail: simple recursive BFS, depth≤3 stays fast on this DB size
    def _recurse(node_id, remaining):
        node = store().get_node(node_id)
        if node is None:
            return None
        result = {k: node.get(k) for k in ("id", "kind", "name", "understanding")}
        if remaining > 0:
            kids = store().children(node_id)
            if kids:
                result["children"] = [_recurse(k["id"], remaining - 1) for k in kids]
        return result
    return _recurse(id, int(depth))


def kg_stat() -> dict:
    db = store().db
    nc = db.execute("SELECT COUNT(*) FROM nodes").fetchone()[0]
    ec = db.execute("SELECT COUNT(*) FROM edges").fetchone()[0]
    kinds = db.execute("SELECT kind, COUNT(*) c FROM nodes GROUP BY kind ORDER BY c DESC").fetchall()
    return {
        "nodes": nc,
        "edges": ec,
        "by_kind": {row[0]: row[1] for row in kinds},
    }


# ── tool registry ─────────────────────────────────────────────────────────────

TOOLS = [
    {
        "name": "kg_search",
        "description": (
            "Search the ARES/EROS/ZEUS homelab knowledge graph: indexed summaries of every "
            "folder in the storage pool, past Claude Code + ChatGPT + claude.ai conversation "
            "history, and installed skills/MCPs/agents. Use this BEFORE grepping the "
            "filesystem, reading memory files one by one, or asking the user for homelab "
            "context (what's on a box, where a service lives, past decisions/incidents) — it "
            "often already has the answer. Tries an exact match first, falls back to a "
            "broader match automatically. Returns matching nodes (id, kind, name, "
            "understanding clipped to 300 chars, path for folders). Call kg_get <id> for "
            "the full text of a hit."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search terms"},
                "limit": {"type": "integer", "default": 8, "description": "Max results"},
            },
            "required": ["query"],
        },
    },
    {
        "name": "kg_get",
        "description": "Get full details of one homelab-kg node by id (from a kg_search result's 'id' field) — its full understanding text, path, and metadata.",
        "inputSchema": {
            "type": "object",
            "properties": {"id": {"type": "string"}},
            "required": ["id"],
        },
    },
    {
        "name": "kg_neighbors",
        "description": "What's connected to a homelab-kg node — parent/child folders, related chats, backup links. Use after kg_search to explore context around a hit.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "id": {"type": "string"},
                "type": {"type": "string", "description": "Edge type filter (optional)"},
            },
            "required": ["id"],
        },
    },
    {
        "name": "kg_tree",
        "description": "Browse a homelab-kg node's subfolder tree with summaries, like a smart 'ls -R' that already knows what's in each folder. Good for 'what's in PROJECTS/X' type questions.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "id": {"type": "string"},
                "depth": {"type": "integer", "default": 2},
            },
            "required": ["id"],
        },
    },
    {
        "name": "kg_stat",
        "description": "Quick health check of the homelab-kg graph itself (total nodes/edges, breakdown by kind) — use to sanity-check the graph is populated, not for answering homelab questions.",
        "inputSchema": {"type": "object", "properties": {}},
    },
]

TOOL_FNS = {
    "kg_search": lambda a: kg_search(**a),
    "kg_get": lambda a: kg_get(**a),
    "kg_neighbors": lambda a: kg_neighbors(**a),
    "kg_tree": lambda a: kg_tree(**a),
    "kg_stat": lambda _: kg_stat(),
}


# ── JSON-RPC 2.0 dispatcher ───────────────────────────────────────────────────

def respond(rid, result=None, error=None):
    msg = {"jsonrpc": "2.0", "id": rid}
    if error is not None:
        msg["error"] = error
    else:
        msg["result"] = result
    sys.stdout.write(json.dumps(msg) + "\n")
    sys.stdout.flush()


def dispatch(msg):
    rid = msg.get("id")
    method = msg.get("method", "")
    params = msg.get("params") or {}

    if method == "initialize":
        respond(rid, {
            "protocolVersion": "2024-11-05",
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "homelab-kg", "version": "1.0.0"},
        })

    elif method == "notifications/initialized":
        pass  # no response for notifications

    elif method == "tools/list":
        respond(rid, {"tools": TOOLS})

    elif method == "tools/call":
        name = params.get("name", "")
        args = params.get("arguments") or {}
        fn = TOOL_FNS.get(name)
        if fn is None:
            respond(rid, error={"code": -32601, "message": f"Unknown tool: {name}"})
            return
        try:
            result = fn(args)
            respond(rid, {
                "content": [{"type": "text", "text": json.dumps(result, default=str)}],
                "isError": False,
            })
        except Exception as e:
            respond(rid, {
                "content": [{"type": "text", "text": f"Error: {e}\n{traceback.format_exc()}"}],
                "isError": True,
            })

    else:
        if rid is not None:
            respond(rid, error={"code": -32601, "message": f"Method not found: {method}"})


def main():
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError as e:
            sys.stdout.write(json.dumps({"jsonrpc": "2.0", "id": None,
                "error": {"code": -32700, "message": f"Parse error: {e}"}}) + "\n")
            sys.stdout.flush()
            continue
        dispatch(msg)


if __name__ == "__main__":
    main()
