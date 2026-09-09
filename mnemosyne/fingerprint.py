import os, hashlib

def folder_fingerprint(path):
    entries = []
    try:
        with os.scandir(path) as it:
            for e in it:
                try:
                    st = e.stat(follow_symlinks=False)
                    is_file = e.is_file(follow_symlinks=False)
                    entries.append((e.name, int(st.st_mtime), st.st_size if is_file else 0))
                except OSError:
                    continue
    except OSError:
        return "0" * 16
    entries.sort()
    h = hashlib.sha256()
    for name, mt, sz in entries:
        h.update(f"{name}\x00{mt}\x00{sz}\n".encode("utf-8", "surrogatepass"))
    return h.hexdigest()[:16]
