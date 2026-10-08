import json

import pytest

from atlas import understanding as U


@pytest.fixture
def calls(monkeypatch):
    seen = []
    def fake_post(url, data):
        seen.append((url, json.loads(data)["model"], json.loads(data).get("options", {})))
        return {"response": "local summary"}
    monkeypatch.setattr(U, "_http_post", fake_post)
    monkeypatch.setattr(U, "openrouter_key", lambda: None)
    monkeypatch.setattr(U, "gpu_on_loan", lambda: False)
    monkeypatch.setattr(U, "photo_vision_busy", lambda: False)
    monkeypatch.setattr(U, "_ares_up", lambda: True)
    return seen


def test_ares_3080_is_first_choice(calls):
    assert U.llm("hi") == "local summary"
    url, model, opts = calls[0]
    assert url == U.LOCAL_LLM_HOST + "/api/generate" and model == U.LOCAL_LLM_MODEL
    assert opts.get("num_ctx") == U.LOCAL_LLM_CTX


def test_gpu_loan_falls_back_to_eros(calls, monkeypatch):
    monkeypatch.setattr(U, "gpu_on_loan", lambda: True)
    U.llm("hi")
    assert calls[0][0] == U.OLLAMA_HOST + "/api/generate" and calls[0][1] == U.OLLAMA_MODEL


def test_ares_down_falls_back_to_eros(calls, monkeypatch):
    monkeypatch.setattr(U, "_ares_up", lambda: False)
    U.llm("hi")
    assert calls[0][0].startswith(U.OLLAMA_HOST)


def test_ares_error_mid_call_falls_back_to_eros(calls, monkeypatch):
    def flaky(url, data):
        calls.append((url, None, None))
        if url.startswith(U.LOCAL_LLM_HOST):
            raise OSError("ares ollama restarting")
        return {"response": "eros summary"}
    monkeypatch.setattr(U, "_http_post", flaky)
    assert U.llm("hi") == "eros summary"
    assert [c[0].split("/api")[0] for c in calls] == [U.LOCAL_LLM_HOST, U.OLLAMA_HOST]


def test_photo_run_pauses_only_eros(calls, monkeypatch):
    monkeypatch.setattr(U, "photo_vision_busy", lambda: True)
    U.llm("hi")
    assert calls[0][0].startswith(U.LOCAL_LLM_HOST)              # ARES unaffected
    monkeypatch.setattr(U, "_ares_up", lambda: False)
    calls.clear()
    assert U.llm("hi") is None and calls == []                   # EROS busy -> skip


def test_digest_size_follows_backend(calls, monkeypatch):
    assert U.digest_chars() == U.LOCAL_DIGEST_CHARS
    monkeypatch.setattr(U, "gpu_on_loan", lambda: True)
    assert U.digest_chars() == U.OLLAMA_DIGEST_CHARS
    monkeypatch.setattr(U, "openrouter_key", lambda: "sk-or-x")
    assert U.digest_chars() == U.OPENROUTER_DIGEST_CHARS
