import os
from atlas.store import Store
from atlas.content import index_content, extract_image, file_fingerprint


def _write(root, rel, data="x"):
    p = os.path.join(root, rel)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    mode = "wb" if isinstance(data, bytes) else "w"
    with open(p, mode) as f:
        f.write(data)
    return p


def test_vault_paths_are_never_indexed(tmp_path):
    root = str(tmp_path / "pool")
    _write(root, "PERSONAL/id/insurance.png", "not a real image, just bytes")
    _write(root, "My Eyes Only/secret.txt", "top secret content")
    _write(root, "PERSONAL/My Eyes Only/nested-secret.md", "also secret")

    called = []
    def spy_tesseract(path):
        called.append(path)
        return "VIN JNKCV64E98M119577"

    st = Store(str(tmp_path / "kg.db"))
    stats = index_content(st, root, box="ARES", tesseract=spy_tesseract, budget=100)

    assert stats["skipped_vault"] >= 2
    for path in called:
        assert "My Eyes Only" not in path
    assert st.get_node("ARES:" + os.path.join(root, "My Eyes Only", "secret.txt")) is None
    assert st.get_node("ARES:" + os.path.join(root, "PERSONAL", "My Eyes Only", "nested-secret.md")) is None
    st.close()


def test_vault_file_matched_directly_is_also_skipped(tmp_path):
    # defense in depth: a file matching the vault pattern even outside a
    # vault-named directory must still never be opened or indexed.
    root = str(tmp_path / "pool")
    _write(root, "weird/vault-secure-notes.txt", "should never be read")
    st = Store(str(tmp_path / "kg.db"))
    stats = index_content(st, root, box="ARES", budget=100)
    assert stats["skipped_vault"] >= 1
    assert st.get_node("ARES:" + os.path.join(root, "weird", "vault-secure-notes.txt")) is None
    st.close()


def test_txt_file_is_indexed_and_searchable(tmp_path):
    root = str(tmp_path / "pool")
    _write(root, "docs/note.txt", "the VIN is JNKCV64E98M119577")
    st = Store(str(tmp_path / "kg.db"))
    stats = index_content(st, root, box="ARES", budget=100)
    assert stats["changed"] == 1
    nid = "ARES:" + os.path.join(root, "docs", "note.txt")
    n = st.get_node(nid)
    assert n["kind"] == "file-content"
    assert "JNKCV64E98M119577" in n["understanding"]
    hits = st.search("JNKCV64E98M119577")
    assert any(h["id"] == nid for h in hits)
    st.close()


def test_unchanged_file_is_skipped_on_second_run(tmp_path):
    root = str(tmp_path / "pool")
    _write(root, "docs/note.txt", "hello world")
    st = Store(str(tmp_path / "kg.db"))
    index_content(st, root, box="ARES", budget=100)
    stats2 = index_content(st, root, box="ARES", budget=100)
    assert stats2["changed"] == 0
    assert stats2["skipped_unchanged"] == 1
    st.close()


def test_changed_file_is_reprocessed(tmp_path):
    root = str(tmp_path / "pool")
    path = _write(root, "docs/note.txt", "hello world")
    st = Store(str(tmp_path / "kg.db"))
    index_content(st, root, box="ARES", budget=100)
    os.utime(path, (0, 0))  # force a different mtime
    with open(path, "w") as f:
        f.write("hello world, updated")
    stats2 = index_content(st, root, box="ARES", budget=100)
    assert stats2["changed"] == 1
    nid = "ARES:" + path
    assert "updated" in st.get_node(nid)["understanding"]
    st.close()


def test_budget_caps_files_processed_per_run(tmp_path):
    root = str(tmp_path / "pool")
    for i in range(5):
        _write(root, f"docs/note{i}.txt", f"content {i}")
    st = Store(str(tmp_path / "kg.db"))
    stats = index_content(st, root, box="ARES", budget=2)
    assert stats["processed"] == 2
    assert stats["changed"] == 2
    assert stats["skipped_budget"] == 3
    st.close()


def test_extract_image_falls_back_to_vision_when_ocr_finds_no_text(tmp_path):
    calls = {"tesseract": 0, "vision": 0}
    def tesseract(path):
        calls["tesseract"] += 1
        return "   \n\x0c"  # tesseract's typical output for a photo with no text
    def vision(path):
        calls["vision"] += 1
        return "A photo of a mountain at sunset."
    text, method = extract_image(str(tmp_path / "img.png"), tesseract, vision)
    assert method == "ollama-vision"
    assert text == "A photo of a mountain at sunset."
    assert calls == {"tesseract": 1, "vision": 1}


def test_extract_image_keeps_ocr_text_without_calling_vision(tmp_path):
    calls = {"vision": 0}
    def tesseract(path):
        return "Vehicle ID No. JNKCV64E98M119577"
    def vision(path):
        calls["vision"] += 1
        return "should not be called"
    text, method = extract_image(str(tmp_path / "img.png"), tesseract, vision)
    assert method == "tesseract"
    assert "JNKCV64E98M119577" in text
    assert calls["vision"] == 0


def test_missing_tesseract_skips_images_without_crashing(tmp_path, monkeypatch):
    monkeypatch.setattr("atlas.content._tesseract_available", lambda: False)
    root = str(tmp_path / "pool")
    _write(root, "photo.png", "fake image bytes")
    st = Store(str(tmp_path / "kg.db"))
    stats = index_content(st, root, box="ARES", budget=100)
    assert stats["errors"] == 0
    assert st.get_node("ARES:" + os.path.join(root, "photo.png")) is None
    st.close()


def test_corrupt_file_is_marked_error_and_walk_continues(tmp_path, monkeypatch):
    monkeypatch.setattr("atlas.content._docx_available", lambda: True)
    root = str(tmp_path / "pool")
    _write(root, "docs/a.docx", "not a real docx")
    _write(root, "docs/b.txt", "still fine")

    def boom_docx(path):
        raise ValueError("corrupt docx")

    st = Store(str(tmp_path / "kg.db"))
    stats = index_content(st, root, box="ARES", budget=100, docx_extract=boom_docx)
    assert stats["errors"] == 1
    assert stats["changed"] == 1  # b.txt still got indexed despite a.docx failing
    a_node = st.get_node("ARES:" + os.path.join(root, "docs", "a.docx"))
    assert a_node["status"] == "error"
    assert "corrupt docx" in a_node["meta"]["error"]
    st.close()


def test_file_fingerprint_changes_with_mtime_and_size():
    class FakeStat:
        def __init__(self, mtime, size):
            self.st_mtime = mtime
            self.st_size = size
    fp1 = file_fingerprint(FakeStat(100, 10))
    fp2 = file_fingerprint(FakeStat(200, 10))
    fp3 = file_fingerprint(FakeStat(100, 10))
    assert fp1 != fp2
    assert fp1 == fp3
