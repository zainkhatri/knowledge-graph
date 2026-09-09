#!/usr/bin/env python3
"""Minimal stdio JSON-RPC 2.0 MCP server for homelab-kg.
Handles: initialize, tools/list, tools/call.
No external deps beyond stdlib + mnemosyne.store.
"""
import json, os, sys, traceback

ROOT = os.path.dirname(__file__)
DB_PATH = os.environ.get("KG_DB", os.path.join(ROOT, "data", "homelab_kg.db"))

sys.path.insert(0, ROOT)
from mnemosyne.store import Store

_store = None

def store():
    global _store
    if _store is None:
        _store = Store(DB_PATH)
    return _store


# ── tool implementations ──────────────────────────────────────────────────────

def kg_search(query: str, limit: int = 15) -> list:
    rows = store().search(query, limit=int(limit))
    return [
        {k: r[k] for k in ("id", "kind", "name", "understanding", "path") if k in r}
        for r in rows
    ]


def kg_get(id: str) -> dict | None:
    return store().get_node(id)


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
        "description": "Full-text search the homelab knowledge graph. Returns matching nodes (id, kind, name, understanding, path).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search terms"},
                "limit": {"type": "integer", "default": 15, "description": "Max results"},
            },
            "required": ["query"],
        },
    },
    {
        "name": "kg_get",
        "description": "Get a single node by id from the homelab knowledge graph.",
        "inputSchema": {
            "type": "object",
            "properties": {"id": {"type": "string"}},
            "required": ["id"],
        },
    },
    {
        "name": "kg_neighbors",
        "description": "Get edges adjacent to a node. Optionally filter by edge type.",
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
        "description": "Return a node's hierarchical children tree up to a given depth.",
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
        "description": "Return node/edge counts and breakdown by kind for the homelab knowledge graph.",
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
