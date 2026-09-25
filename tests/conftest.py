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
