import pytest
from atlas.redact import redact

SECRETS = [
    "key is " + "sk-or" + "-v1-" + "fake2e5a9f0c1b2d" * 4,
    "export ANTHROPIC_API_KEY=" + "sk-" + "ant-api03-AbCdEf123456GhIjKl789012MnOpQr",
    "token " + "gh" + "p_A1b2C3d4E5f6G7h8I9j0K1l2M3n4O5p6Q7r8",
    "slack " + "xo" + "xb-123456789012-1234567890123-AbCdEfGhIjKlMnOpQrStUvWx",
    "aws AKIAIOSFODNN7EXAMPLE",
    "google " + "AI" + "zaSyA-1234567890abcdefghijklmnopqrstu",
    "curl -H 'Authorization: Bearer abc123DEF456ghi789JKL'",
    "jwt " + "eyJ" + "hbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.dozjgNryP4J3jVmNHl0w5N_XgL0n3I9PlFUP0THsR8U",
    "ARES_PASSWORD=hunter2correcthorse",
    '"api_key": "live_9f8e7d6c5b4a"',
    "SSO_SECRET: 'q8w7e6r5t4y3'",
    "postgres://admin:s3cretPw@db.local:5432/app",
    "-----BEGIN OPENSSH " + "PRIVATE KEY-----\nb3BlbnNzaC1rZXktdjEAAAAA\n-----END OPENSSH " + "PRIVATE KEY-----",
    "random Xk9vQ2mP7rT4wZ8nB3cF6hJ1sD5gL0aE",
]
SECRET_BITS = ["fake2e5a9f0c1b2d", "AbCdEf123456", "A1b2C3d4E5f6", "123456789012-1234567890123",
               "AKIAIOSFODNN7EXAMPLE", "AIzaSyA-1234567890", "abc123DEF456", "dozjgNryP4J3",
               "hunter2correcthorse", "live_9f8e7d6c5b4a", "q8w7e6r5t4y3", "s3cretPw",
               "b3BlbnNzaC1rZXktdjEAAAAA", "Xk9vQ2mP7rT4wZ8nB3cF6hJ1sD5gL0aE"]


@pytest.mark.parametrize("text,bit", list(zip(SECRETS, SECRET_BITS)))
def test_secret_removed(text, bit):
    out = redact(text)
    assert bit not in out
    assert "[REDACTED]" in out


KEEP = [
    "/mnt/nvme/PROMETHEUS/PROJECTS/ARES-DASHBOARD/system/kg-nightly.sh",
    "commit a1b2c3d4e5f60718293a4b5c6d7e8f9012345678",
    "session 5b85dd5a-711e-4b1d-b08b-d462c6917373",
    "sha256 9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08",
    "restart with pct exec 101 -- systemctl restart ares",
    "the password prompt appeared twice and the token refresh failed",
    "claude-opus-5-5 and google/gemini-2.5-flash-lite",
]


@pytest.mark.parametrize("text", KEEP)
def test_benign_unchanged(text):
    assert redact(text) == text


def test_empty():
    assert redact("") == ""
    assert redact(None) == ""
