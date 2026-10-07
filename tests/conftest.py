import pytest


@pytest.fixture(autouse=True)
def _no_real_gpu_loan(monkeypatch, tmp_path):
    """Tests must not depend on the live .gpu-on-loan flag (set whenever VM 200 runs).
    Point both modules at a path that never exists; tests that exercise loan behaviour
    monkeypatch gpu_on_loan themselves, which still wins."""
    missing = str(tmp_path / "no-gpu-on-loan")
    monkeypatch.setattr("atlas.understanding.GPU_LOAN_FLAG", missing)
    monkeypatch.setattr("atlas.embeddings.GPU_LOAN_FLAG", missing)


@pytest.fixture(autouse=True)
def _no_real_network(monkeypatch):
    """No test may reach live Ollama or OpenRouter (slow, flaky, and OpenRouter costs money).
    Tests that need a model response inject `http=` or monkeypatch the summarizer."""
    def offline(*a, **k):
        raise OSError("network disabled in tests")
    monkeypatch.setattr("atlas.understanding._http_post", offline)
    monkeypatch.setattr("atlas.understanding._openrouter_post", offline)
    monkeypatch.setattr("atlas.embeddings._http_post", offline)


@pytest.fixture(autouse=True)
def _private_budget(monkeypatch, tmp_path):
    """Never touch the live daily budget/alert files; generous caps unless a test lowers them."""
    from atlas import budget as B
    monkeypatch.setattr(B, "STATE", str(tmp_path / "llm-budget.json"))
    monkeypatch.setattr(B, "ALERT_FILE", str(tmp_path / "atlas-ALERT"))
    monkeypatch.setattr(B, "DAILY_CALLS", 10_000)
    monkeypatch.setattr(B, "DAILY_USD", 100.0)


@pytest.fixture(autouse=True)
def _no_real_photo_busy_flag(monkeypatch, tmp_path):
    """The live kg-photo-vision busy flag must not leak into tests (it pauses Ollama)."""
    monkeypatch.setattr("atlas.understanding.PHOTO_ACTIVE_FLAG", str(tmp_path / "no-photo-run"))
