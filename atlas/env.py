"""Index this box's Claude environment — skills + MCP servers — into the knowledge graph.

So a fresh Claude account / new machine can ask the local model "what skills and MCPs do I
use?" and get the list plus how to re-install each. Skills come from ~/.claude/skills +
plugins (SKILL.md frontmatter); MCPs come from ~/.claude.json (with the re-add command).
"""
import os, glob, json, re, subprocess

# short purpose blurbs for MCPs whose command isn't self-describing
MCP_PURPOSE = {
    "homelab-kg": "The homelab knowledge graph — kg_search/get/neighbors/tree/stat over the filesystem + every past Claude/ChatGPT conversation.",
    "ollama": "Local LLM tools (local_llm to summarize/compress, describe_image for screenshots) via the box's Ollama.",
    "context7": "Live, version-accurate library/framework docs on demand.",
    "playwright": "Headless browser automation (navigate/click/screenshot), incl. the CDP bridge to the Mac.",
    "sequential-thinking": "Structured step-by-step reasoning scratchpad.",
    "vercel-grep": "Search code across Vercel/GitHub.",
    "ruflo": "Claude Flow (ruvnet) — multi-agent swarm orchestration meta-harness, ~314 tools.",
}


def _frontmatter(path):
    name = desc = None
    try:
        txt = open(path, errors="ignore").read(4000)
        m = re.search(r"^---\s*(.*?)\s*---", txt, re.S | re.M)
        block = m.group(1) if m else txt
        nm = re.search(r"^name:\s*(.+)$", block, re.M)
        dm = re.search(r"^description:\s*(.+)$", block, re.M)
        name = (nm.group(1).strip() if nm else os.path.basename(os.path.dirname(path)))
        desc = dm.group(1).strip() if dm else ""
    except Exception:
        pass
    return name, desc


def index_skills(store, box="ARES", claude_home="/root/.claude"):
    hub = f"{box}:skills"
    store.upsert_node({"id": hub, "box": box, "kind": "folder", "path": f"/{box}/SKILLS",
        "name": "Claude Skills",
        "understanding": "Claude Code skills installed on this box — invokable capabilities."})
    n = 0
    paths = glob.glob(f"{claude_home}/skills/*/SKILL.md") + \
            glob.glob(f"{claude_home}/plugins/**/SKILL.md", recursive=True)
    for path in paths:
        name, desc = _frontmatter(path)
        if not name:
            continue
        src = "plugin" if "/plugins/" in path else "user"
        nid = f"{box}:skill/{name}"
        store.upsert_node({"id": nid, "box": box, "kind": "skill", "path": path,
            "name": name, "understanding": (desc or name)[:700], "status": "live",
            "meta": {"invoke": f"/{name}", "source": src}})
        store.add_edge(hub, nid, "contains")
        n += 1
    store.db.commit()
    return {"skills": n}


def index_mcps(store, box="ARES", claude_json="/root/.claude.json"):
    hub = f"{box}:mcps"
    store.upsert_node({"id": hub, "box": box, "kind": "folder", "path": f"/{box}/MCPS",
        "name": "MCP Servers",
        "understanding": "Model Context Protocol servers configured on this box — tools available to Claude and the local model."})
    try:
        d = json.load(open(claude_json))
    except Exception:
        return {"mcps": 0, "error": "no config"}
    seen = {}
    def walk(o):
        if isinstance(o, dict):
            for k, v in o.items():
                if k == "mcpServers" and isinstance(v, dict):
                    for name, cfg in v.items():
                        seen.setdefault(name, cfg)
                else:
                    walk(v)
    walk(d)
    # which claude.ai account connectors are actually connected (live)
    connected = set()
    try:
        out = subprocess.run(["claude", "mcp", "list"], capture_output=True, text=True, timeout=25).stdout
        for line in out.splitlines():
            if "✔" in line or "Connected" in line:
                connected.add(line.split(":")[0].strip())
    except Exception:
        pass
    n = 0
    for name, cfg in seen.items():
        cmd = cfg.get("command", "")
        args = " ".join(cfg.get("args", [])) if isinstance(cfg.get("args"), list) else ""
        url = cfg.get("url", "")
        how = (f"{cmd} {args}".strip() or url)
        install = (f"claude mcp add -s user {name} {how}" if cmd
                   else f"claude mcp add -s user --transport http {name} {url}")
        purpose = MCP_PURPOSE.get(name, "")
        und = (f"{purpose} " if purpose else "") + f"Re-add on a new machine: `{install}`"
        nid = f"{box}:mcp/{name}"
        store.upsert_node({"id": nid, "box": box, "kind": "mcp", "path": claude_json,
            "name": name, "understanding": und[:700], "status": "live",
            "meta": {"how": how, "install": install, "purpose": purpose}})
        store.add_edge(hub, nid, "contains")
        n += 1
    # note the account connectors you actually use (re-enable in claude.ai settings on a new account)
    acct = sorted(c for c in connected if c.startswith("claude.ai "))
    if acct:
        nid = f"{box}:mcp/_account-connectors"
        store.upsert_node({"id": nid, "box": box, "kind": "mcp", "path": claude_json,
            "name": "claude.ai account connectors (in use)",
            "understanding": "Account-level connectors currently CONNECTED (re-enable these in claude.ai → Settings → Connectors on a new account): "
                             + ", ".join(a.replace("claude.ai ", "") for a in acct), "status": "live"})
        store.add_edge(hub, nid, "contains")
        n += 1
    store.db.commit()
    return {"mcps": n, "connected_account_connectors": len(acct)}


def index_agents(store, box="ARES", claude_home="/root/.claude"):
    hub = f"{box}:agents"
    store.upsert_node({"id": hub, "box": box, "kind": "folder", "path": f"/{box}/AGENTS",
        "name": "Sub-Agents",
        "understanding": "Custom Claude sub-agent types available on this box (incl. ruflo swarm/SPARC agents)."})
    n = 0
    paths = glob.glob(f"{claude_home}/agents/*.md") + \
            glob.glob(f"{claude_home}/plugins/**/agents/*.md", recursive=True)
    for path in paths:
        name, desc = _frontmatter(path)
        if not name:
            continue
        nid = f"{box}:agent/{name}"
        store.upsert_node({"id": nid, "box": box, "kind": "agent", "path": path,
            "name": name, "understanding": (desc or name)[:700], "status": "live",
            "meta": {"source": "agent"}})
        store.add_edge(hub, nid, "contains")
        n += 1
    store.db.commit()
    return {"agents": n}


def index_env(store, box="ARES", homes=None, claude_json="/root/.claude.json"):
    homes = homes or ["/root/.claude", "/mnt/nvme/PROMETHEUS/.claude"]
    tot = {"skills": 0, "agents": 0}
    for h in homes:
        if not os.path.isdir(h):
            continue
        tot["skills"] += index_skills(store, box, h).get("skills", 0)
        tot["agents"] += index_agents(store, box, h).get("agents", 0)
    tot.update(index_mcps(store, box, claude_json))
    return tot
