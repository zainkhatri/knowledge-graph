import os

LEAF_MAX = 40
MAX_DEPTH = 8
MAX_NODES = 20000
MARKERS = {".git", "README.md", "README", "package.json", "pyproject.toml",
           "requirements.txt", "docker-compose.yml", "Cargo.toml", "go.mod"}

def node_id(box, path):
    return f"{box}:{path}"

def _list(path):
    dirs, files = [], []
    try:
        with os.scandir(path) as it:
            for e in it:
                try:
                    if e.is_dir(follow_symlinks=False):
                        dirs.append(e.name)
                    elif e.is_file(follow_symlinks=False):
                        files.append(e.name)
                except OSError:
                    continue
    except OSError:
        pass
    return sorted(dirs), sorted(files)

def _ext_hist(files):
    h = {}
    for f in files:
        ext = os.path.splitext(f)[1].lower() or "(none)"
        h[ext] = h.get(ext, 0) + 1
    return h

def has_marker(files, dirs):
    return bool((set(files) | set(dirs)) & MARKERS)

def classify(path, dirs, files):
    if has_marker(files, dirs):
        return "branch", "project"
    if dirs:
        return "branch", "folder"
    if len(files) > LEAF_MAX:
        return "cluster", "file-cluster"
    return "branch", "folder"

def walk(root, box, vault_pred=None, fp=None):
    from .fingerprint import folder_fingerprint
    fpf = fp or folder_fingerprint
    root = os.path.abspath(root)
    count = 0
    stack = [(root, None, 0)]
    while stack:
        path, parent, depth = stack.pop()
        if count >= MAX_NODES or depth > MAX_DEPTH:
            continue
        name = os.path.basename(path) or path
        nid = node_id(box, path)
        try:
            st = os.stat(path)
        except OSError:
            continue
        if vault_pred and vault_pred(path):
            yield {"id": nid, "box": box, "kind": "vault", "path": path, "name": name,
                   "fingerprint": "vault", "size": 0, "mtime": int(st.st_mtime),
                   "meta": {}, "parent": parent,
                   "understanding": "Encrypted vault — contents not indexed."}
            count += 1
            continue
        dirs, files = _list(path)
        flag, kind = classify(path, dirs, files)
        yield {"id": nid, "box": box, "kind": kind, "path": path, "name": name,
               "fingerprint": fpf(path), "size": 0, "mtime": int(st.st_mtime),
               "meta": {"n_dirs": len(dirs), "n_files": len(files), "ext": _ext_hist(files)},
               "parent": parent}
        count += 1
        if flag == "branch" and depth < MAX_DEPTH:
            for d in dirs:
                stack.append((os.path.join(path, d), nid, depth + 1))
