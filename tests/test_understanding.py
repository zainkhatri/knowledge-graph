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
