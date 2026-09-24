from atlas import understanding as U

NODE = {"kind": "project", "path": "/x/bdr", "name": "bdr",
        "meta": {"n_dirs": 3, "n_files": 12, "ext": {".js": 8, ".json": 4}}}

def test_generate_uses_injected_http():
    seen = {}
    def fake_http(url, data):
        seen["url"] = url; seen["data"] = data
        return {"response": "  Business BDR outreach stack for FAI/FCSF.  "}
    out = U.generate(NODE, ["FAI sender", "FCSF sender"], http=fake_http)
    assert out == "Business BDR outreach stack for FAI/FCSF."
    assert seen["url"].endswith("/api/generate")
    assert b"bdr" in seen["data"]

def test_vault_never_calls_model():
    called = {"n": 0}
    def fake_http(url, data):
        called["n"] += 1; return {"response": "SHOULD NOT HAPPEN"}
    out = U.generate({"kind": "vault", "path": "/v", "name": "v", "meta": {}}, http=fake_http)
    assert out == "Encrypted vault — contents not indexed."
    assert called["n"] == 0

def test_http_failure_returns_none():
    def boom(url, data):
        raise RuntimeError("ollama down")
    assert U.generate(NODE, [], http=boom) is None

def test_gpu_loan_skips(monkeypatch):
    monkeypatch.setattr(U, "gpu_on_loan", lambda: True)
    called = {"n": 0}
    def fake_http(url, data):
        called["n"] += 1; return {"response": "x"}
    assert U.generate(NODE, [], http=fake_http) is None
    assert called["n"] == 0


def _or_http(seen, reply="Fixed the Caddy config on ARES."):
    def fake(key, body):
        seen["key"] = key; seen["body"] = body
        return {"choices": [{"message": {"content": "  " + reply + "  "}}]}
    return fake


def test_summarize_session_uses_openrouter_and_redacts(monkeypatch):
    monkeypatch.setattr(U, "openrouter_key", lambda: "sk-or-test")
    monkeypatch.setattr(U, "gpu_on_loan", lambda: True)       # remote call ignores the GPU loan
    seen = {}
    out = U.summarize_session("USER: set ARES_PASSWORD=hunter2correcthorse and fix caddy",
                              title="Caddy", cwd="/x", box="ARES", http=_or_http(seen))
    assert out == "Fixed the Caddy config on ARES."
    prompt = seen["body"]["messages"][0]["content"]
    assert "hunter2correcthorse" not in prompt and "[REDACTED]" in prompt
    assert "fix caddy" in prompt and "Box: ARES" in prompt
    assert seen["body"]["model"] == U.SUMMARY_MODEL


def test_summarize_chat_redacts_asks(monkeypatch):
    monkeypatch.setattr(U, "openrouter_key", lambda: "sk-or-test")
    seen = {}
    U.summarize_chat(["use key " + "sk-" + "ant-api03-AbCdEf123456GhIjKl789012"], http=_or_http(seen))
    assert "AbCdEf123456" not in seen["body"]["messages"][0]["content"]


def test_out_of_credit_propagates(monkeypatch):
    monkeypatch.setattr(U, "openrouter_key", lambda: "sk-or-test")
    def broke(key, body):
        raise U.OutOfCredit("402")
    import pytest
    with pytest.raises(U.OutOfCredit):
        U.summarize_session("USER: hi", http=broke)


def test_transient_failure_returns_none(monkeypatch):
    monkeypatch.setattr(U, "openrouter_key", lambda: "sk-or-test")
    calls = {"n": 0}
    def flaky(key, body):
        calls["n"] += 1; raise RuntimeError("503")
    assert U.summarize_session("USER: hi", http=flaky) is None
    assert calls["n"] == 3
