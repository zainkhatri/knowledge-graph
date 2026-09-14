# ATLAS

A knowledge graph for my homelab. I got tired of agents burning tokens re-reading the same
folders every session, so ATLAS walks the filesystem once, has a local model (Ollama) write
a one-line summary of each folder, and stores it all in SQLite. Any agent I run can then
query the graph (`kg_search`, `kg_get`, `kg_neighbors`, `kg_tree`) instead of `ls`/`cat`-ing
its way through thousands of files. It's exposed as an MCP server so this works from any
agent, not just one tool.

A few things worth pointing out:

- **Summaries are generated, not written by hand.** Each node's understanding comes from
  a local model looking at the folder's structure — no manual note-taking.
- **Only changed folders get re-summarized.** A cheap structural fingerprint per folder
  means re-walking the tree is fast — it skips anything that hasn't changed.
- **Multiple boxes merge into one graph.** I run this across 3 machines; each builds its
  own graph locally, and `kg merge` folds one box's graph into a central store.
- It also indexes Claude Code session transcripts and chat exports as graph nodes, so past
  work is searchable alongside the filesystem, and it can index installed skills/MCP
  servers/agents so a box's tooling is part of the graph too.
- The MCP server itself is stdlib-only — no framework, just a ~200-line stdio JSON-RPC 2.0
  server over SQLite.

## Layout

- `atlas/walker.py` — the filesystem walker (adaptive depth, skips node_modules/.git/etc)
- `atlas/fingerprint.py` — change detection
- `atlas/understanding.py` — Ollama summarization
- `atlas/store.py` — SQLite + FTS5 storage, WAL mode
- `atlas/link.py` — merging graphs across boxes
- `atlas/chats.py`, `atlas/claude_web.py`, `atlas/gpt.py` — indexing chat history
- `atlas/env.py` — indexing installed skills/MCP servers/agents
- `mcp_server.py` — the MCP server itself

## CLI

```
kg walk <root>                # walk a directory into the graph
kg index-chats                # index Claude Code session transcripts
kg index-claude-web <export>  # index a Claude.ai conversations.json export
kg index-gpt <archive>        # index a ChatGPT export
kg index-env                  # index installed skills/MCP servers/agents
kg merge <other-db>           # merge another box's graph into this one
kg search "<query>"           # full-text search over the graph
```

## Tests

```
pytest
```
