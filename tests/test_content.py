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


def test_dot_vault_directory_is_never_indexed(tmp_path):
    # Regression test: PHOTOS_ROOT/.vault/ is the REAL, actual vault storage
    # path used by ARES-DASHBOARD (app.py's _VAULT_ORIGINALS_DIR) — a
    # dot-prefixed hidden directory, not one of the literal substrings
    # ("my eyes only", "/vault", "vault-secure") the predicate originally
    # checked for. "/.vault/" does NOT contain the substring "/vault" (the
    # dot sits between the slash and "vault"), so the old predicate walked
    # straight into it. 3 real photos were OCR'd into the graph before this
    # was caught and purged (2026-09-27) — this test locks in the fix.
    root = str(tmp_path / "pool")
    _write(root, "PHOTOS/.vault/iPhone/2026/09/IMG_5390.JPG", "not a real image")
    _write(root, "PHOTOS/G7X/IMG_0001.JPG", "not a real image either")

    called = []
    def spy_tesseract(path):
        called.append(path)
        return "should never see vault content"

    st = Store(str(tmp_path / "kg.db"))
    stats = index_content(st, root, box="ARES", tesseract=spy_tesseract, budget=100)

    assert stats["skipped_vault"] >= 1
    for path in called:
        assert ".vault" not in path
    assert st.get_node("ARES:" + os.path.join(root, "PHOTOS", ".vault", "iPhone", "2026", "09", "IMG_5390.JPG")) is None
    # the sibling non-vault photo still gets processed normally
    assert st.get_node("ARES:" + os.path.join(root, "PHOTOS", "G7X", "IMG_0001.JPG")) is not None
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


def test_session_archive_is_never_indexed(tmp_path):
    # Raw session dumps (tool-results/*.txt) duplicate the summarized chat nodes
    # and were 1 in 4 of all top-8 search hits before this exclusion (2026-09-28).
    root = str(tmp_path / "pool")
    _write(root, "PERSONAL/CLAUDE-CODE-SESSIONS/ARES/proj/abc/tool-results/b1.txt", "grep output noise")
    _write(root, "PERSONAL/notes.txt", "a real note about the backup chain")

    st = Store(str(tmp_path / "kg.db"))
    index_content(st, root, box="ARES", budget=100)

    assert st.get_node("ARES:" + os.path.join(root, "PERSONAL/notes.txt")) is not None
    dump = os.path.join(root, "PERSONAL/CLAUDE-CODE-SESSIONS/ARES/proj/abc/tool-results/b1.txt")
    assert st.get_node("ARES:" + dump) is None
    st.close()


def test_near_empty_ocr_is_kept_as_marker_but_not_searchable(tmp_path):
    root = str(tmp_path / "pool")
    _write(root, "PHOTOS/IMG_0001.JPG", "not a real image")
    _write(root, "PHOTOS/IMG_0002.JPG", "not a real image")
    ocr = {"IMG_0001.JPG": "a i\n<> x qwzt", "IMG_0002.JPG": "zebra crossing sign downtown"}
    calls = []
    def fake_tesseract(path):
        calls.append(path)
        return ocr[os.path.basename(path)]

    st = Store(str(tmp_path / "kg.db"))
    index_content(st, root, box="ARES", tesseract=fake_tesseract, vision=lambda p: None, budget=100)

    empty = st.get_node("ARES:" + os.path.join(root, "PHOTOS/IMG_0001.JPG"))
    assert empty is not None and empty["status"] == "empty"
    assert [n["name"] for n in st.search("qwzt")] == []
    assert [n["name"] for n in st.search("zebra crossing")] == ["IMG_0002.JPG"]

    calls.clear()
    index_content(st, root, box="ARES", tesseract=fake_tesseract, vision=lambda p: None, budget=100)
    assert calls == []          # fingerprint marker means no re-OCR next night
    st.close()


def test_has_text_keeps_ids_and_real_words():
    from atlas.content import has_text
    assert has_text("VIN JNKCV64E98M119577")
    assert has_text("Soy El leader de Santos")
    assert not has_text("Mom\n6/2/21, 4:03 PM")
    assert not has_text("¢ & & & & <> ¢€> €>")
    assert not has_text("")
