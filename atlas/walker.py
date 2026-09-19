import os
from .fingerprint import folder_fingerprint

LEAF_MAX = int(os.getenv("KG_LEAF_MAX", "40"))
MAX_DEPTH = int(os.getenv("KG_MAX_DEPTH", "8"))   # overview indexes use a shallow cap (e.g. 4)
MAX_NODES = int(os.getenv("KG_MAX_NODES", "20000"))
MARKERS = {".git", "README.md", "README", "package.json", "pyproject.toml",
           "requirements.txt", "docker-compose.yml", "Cargo.toml", "go.mod"}
# Dev/build noise we never descend into or make nodes for — they explode the
# graph with thousands of meaningless folders (node_modules, .git internals…).
IGNORE_DIRS = {"node_modules", ".git", ".venv", "venv", "__pycache__",
               ".pytest_cache", ".mypy_cache", ".tox", "dist", "build", ".next",
               ".nuxt", ".output", ".turbo", ".cache", ".parcel-cache", "coverage",
               ".idea", ".vscode", "target", "vendor", "bower_components", ".svn",
               ".terraform", ".gradle", "__snapshots__", ".ipynb_checkpoints",
               # Postgres data directories — engine internals, not content. Named
               # PGDATA dirs on the boxes we index, plus their pg_* subfolders in
               # case a data dir is ever reached via a different parent path.
               "ibt-db-data", "pg_serial", "pg_notify", "pg_twophase", "pg_multixact",
               "pg_xact", "pg_logical", "pg_replslot", "pg_snapshots", "pg_stat",
               "pg_stat_tmp", "pg_tblspc", "pg_dynshmem", "pg_commit_ts",
               "pg_subtrans", "pg_wal", "pg_xlog",
               # Filesystem/OS reserved dirs — never real content
               "lost+found", ".lost+found"}

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
               "fingerprint": fpf(path), "size": st.st_size, "mtime": int(st.st_mtime),
               "meta": {"n_dirs": len(dirs), "n_files": len(files), "ext": _ext_hist(files)},
               "parent": parent}
        count += 1
        if flag == "branch":
            for d in dirs:
                if d in IGNORE_DIRS:
                    continue
                stack.append((os.path.join(path, d), nid, depth + 1))

def default_vault_pred():
    pats = [p.strip().lower() for p in
            os.getenv("VAULT_PATHS", "my eyes only,/vault,vault-secure").split(",") if p.strip()]
    def pred(path):
        pl = path.lower()
        return any(p in pl for p in pats)
    return pred
