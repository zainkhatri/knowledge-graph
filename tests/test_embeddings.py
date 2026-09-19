from atlas import embeddings as E

def test_embed_uses_injected_http():
    seen = {}
    def fake_http(url, data):
        seen["url"] = url; seen["data"] = data
        return {"embedding": [0.1, 0.2, 0.3]}
    out = E.embed("photos folder with 4000 images", http=fake_http)
    assert out == [0.1, 0.2, 0.3]
    assert seen["url"].endswith("/api/embeddings")
    assert b"nomic-embed-text" in seen["data"]

def test_empty_text_returns_none_without_calling_http():
    called = {"n": 0}
    def fake_http(url, data):
        called["n"] += 1; return {"embedding": [1.0]}
    assert E.embed("", http=fake_http) is None
    assert E.embed("   ", http=fake_http) is None
    assert called["n"] == 0

def test_http_failure_returns_none():
    def boom(url, data):
        raise RuntimeError("ollama down")
    assert E.embed("some text", http=boom) is None

def test_missing_embedding_key_returns_none():
    def fake_http(url, data):
        return {}
    assert E.embed("some text", http=fake_http) is None

def test_gpu_loan_skips(monkeypatch):
    monkeypatch.setattr(E, "gpu_on_loan", lambda: True)
    called = {"n": 0}
    def fake_http(url, data):
        called["n"] += 1; return {"embedding": [1.0]}
    assert E.embed("some text", http=fake_http) is None
    assert called["n"] == 0
