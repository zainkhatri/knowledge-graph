"""Hard daily cap on LLM calls and OpenRouter spend, shared by every atlas process.

Why: 2026-10-01..05 a path ping-pong (ARES-FAILOVER symlink aliases) made the hourly
sync re-summarize ~3,300 sessions per run; ~306K OpenRouter calls burned the key's $50
limit in five days with no alert. Per-run budgets cannot catch a loop that repeats every
hour, so the cap is per DAY and lives in one state file (flock'd: the hourly sync, the
nightly job and the summary worker threads all share it).

reserve() before a call; record_cost(usd) after an OpenRouter call. Past either cap,
reserve() returns False and writes an alert file once per day.
"""
import fcntl, json, os, time

STATE = os.getenv("KG_BUDGET_FILE", "/var/lib/atlas/llm-budget.json")
ALERT_FILE = os.getenv("KG_ALERT_FILE", "/mnt/nvme/PROMETHEUS/INFRA/status/atlas-ALERT")
DAILY_CALLS = int(os.getenv("KG_DAILY_CALLS", "300"))      # all LLM calls (OpenRouter + Ollama)
DAILY_USD = float(os.getenv("KG_DAILY_USD", "0.25"))        # OpenRouter spend


def _today():
    return time.strftime("%Y-%m-%d")


def alert(msg):
    """Write a loud, dated alert file (and stderr) instead of failing quietly."""
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} atlas: {msg}\n"
    try:
        os.makedirs(os.path.dirname(ALERT_FILE), exist_ok=True)
        with open(ALERT_FILE, "a") as f:
            f.write(line)
    except OSError:
        pass
    print("ALERT " + line.strip(), flush=True)


def _update(fn):
    """Run fn(state) under an exclusive lock; state resets at local midnight."""
    os.makedirs(os.path.dirname(STATE) or ".", exist_ok=True)
    with open(STATE, "a+") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.seek(0)
        try:
            st = json.loads(f.read() or "{}")
        except ValueError:
            st = {}
        if st.get("date") != _today():
            st = {"date": _today(), "calls": 0, "usd": 0.0, "alerted": False}
        out = fn(st)
        f.seek(0); f.truncate(); f.write(json.dumps(st))
        return out


def reserve():
    def go(st):
        if st["calls"] >= DAILY_CALLS or st["usd"] >= DAILY_USD:
            first = not st.get("alerted")
            st["alerted"] = True
            return False, first, dict(st)
        st["calls"] += 1
        return True, False, None
    ok, first, snap = _update(go)
    if first:
        alert(f"daily cap hit ({snap['calls']}/{DAILY_CALLS} calls, ${snap['usd']:.3f}/"
              f"${DAILY_USD:.2f}); LLM summaries paused until tomorrow")
    return ok


def record_cost(usd):
    if usd and usd > 0:
        _update(lambda st: st.__setitem__("usd", round(st["usd"] + float(usd), 6)))


def status():
    return _update(lambda st: {"date": st["date"], "calls": st["calls"], "usd": st["usd"],
                               "calls_cap": DAILY_CALLS, "usd_cap": DAILY_USD})
