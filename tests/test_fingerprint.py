import os, time
from mnemosyne.fingerprint import folder_fingerprint

def test_fingerprint_changes_on_child_add(tmp_path):
    d = tmp_path / "proj"; d.mkdir()
    (d / "a.txt").write_text("hi")
    fp1 = folder_fingerprint(str(d))
    assert len(fp1) == 16
    (d / "b.txt").write_text("yo")
    fp2 = folder_fingerprint(str(d))
    assert fp1 != fp2

def test_fingerprint_stable_when_unchanged(tmp_path):
    d = tmp_path / "proj"; d.mkdir(); (d / "a.txt").write_text("hi")
    assert folder_fingerprint(str(d)) == folder_fingerprint(str(d))

def test_fingerprint_missing_path_is_safe():
    assert folder_fingerprint("/no/such/path/xyz") == "0" * 16
