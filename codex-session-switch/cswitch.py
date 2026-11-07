#!/usr/bin/env python3.12
"""Move one local Codex thread between independent CODEX_HOME directories."""
from __future__ import annotations

import argparse
import base64
import contextlib
import datetime as dt
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import tomllib
import uuid

VERSION = "1.0.0"
UUID = r"[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}"
STATE_TABLES = ("threads", "thread_dynamic_tools", "thread_attachments")
HISTORY_TABLES = ("thread_turns", "thread_items", "thread_history_projection_state", "thread_realtime_items")
GOAL_TABLES = ("thread_goals", "thread_goal_continuation_deferrals")


class Error(Exception):
    pass


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd, tmp = tempfile.mkstemp(prefix=".cswitch-", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def read_json(path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def encode(v):
    if isinstance(v, bytes):
        return {"$bytes": base64.b64encode(v).decode()}
    return v


def decode(v):
    if isinstance(v, dict) and set(v) == {"$bytes"}:
        return base64.b64decode(v["$bytes"])
    return v


def config(home):
    p = home / "config.toml"
    return tomllib.loads(p.read_text()) if p.exists() else {}


def db_path(home, kind):
    # Custom DB locations must be understood explicitly rather than silently ignored.
    if config(home).get("sqlite_home"):
        raise Error(f"{home} 设置了 sqlite_home；当前工具不支持自定义数据库位置。")
    found = list(home.glob(kind + "_[0-9]*.sqlite"))
    if len(found) > 1:
        raise Error(f"{home} 有多个 {kind} 数据库版本，需要检查后再迁移。")
    return found[0] if found else None


def db_read(path):
    c = sqlite3.connect(path.as_uri() + "?mode=ro", uri=True, timeout=10)
    c.row_factory = sqlite3.Row
    return c


def tables(c):
    return {r[0] for r in c.execute("SELECT name FROM sqlite_master WHERE type='table'")}


def thread_row(home, tid):
    p = db_path(home, "state")
    if not p:
        return None
    with contextlib.closing(db_read(p)) as c:
        if "threads" not in tables(c):
            return None
        row = c.execute("SELECT * FROM threads WHERE id=?", (tid,)).fetchone()
        return dict(row) if row else None


def list_rows(home):
    p = db_path(home, "state")
    if not p:
        return []
    with contextlib.closing(db_read(p)) as c:
        if "threads" not in tables(c):
            return []
        return [dict(r) for r in c.execute("SELECT * FROM threads WHERE archived=0 ORDER BY updated_at DESC")]


def managed_path(home, path):
    p = Path(path)
    if not p.is_absolute():
        p = home / p
    try:
        rel = p.resolve().relative_to(home.resolve())
    except ValueError:
        raise Error(f"会话记录路径不在其 CODEX_HOME 内：{p}")
    if not rel.parts or rel.parts[0] not in ("sessions", "archived_sessions"):
        raise Error(f"不支持的会话记录路径：{p}")
    return rel


def existing_rollout(home, path):
    rel = managed_path(home, path)
    p = home / rel
    if p.exists():
        return p
    if p.suffix == ".jsonl" and Path(str(p) + ".zst").exists():
        return Path(str(p) + ".zst")
    if p.suffix == ".zst" and p.with_suffix("").exists():
        return p.with_suffix("")
    raise Error(f"缺少会话记录：{p}")


def rollout_bytes(path):
    if path.suffix != ".zst":
        return path.read_bytes()
    zstd = shutil.which("zstd")
    if not zstd:
        raise Error("读取压缩会话需要 zstd（brew install zstd）。")
    r = subprocess.run([zstd, "-q", "-d", "-c", str(path)], capture_output=True)
    if r.returncode:
        raise Error(f"无法解压会话：{path}")
    return r.stdout


def header(data, path):
    try:
        item = json.loads(data.split(b"\n", 1)[0])
        if item["type"] != "session_meta":
            raise ValueError("missing session_meta")
        uuid.UUID(item["payload"]["id"])
        return item["payload"]
    except (ValueError, KeyError, IndexError):
        raise Error(f"会话头记录无效：{path}")


def rollout_id(path):
    name = path.name.removesuffix(".zst").removesuffix(".jsonl")
    matches = re.findall(UUID, name)
    return matches[-1] if matches else None


def find_ancestor(home, rid):
    # Rollout IDs, unlike thread IDs, remain unique after revert/fork.
    found = []
    for folder in ("sessions", "archived_sessions"):
        for p in (home / folder).rglob("*" + rid + "*.jsonl*"):
            if p.is_file() and rollout_id(p) == rid:
                found.append(p)
    plain = {p.with_suffix("") if p.suffix == ".zst" else p for p in found}
    if len(plain) != 1:
        raise Error(f"无法唯一定位前序记录 {rid}（找到 {len(plain)} 份）。")
    p = next(iter(plain))
    return p if p.exists() else Path(str(p) + ".zst")


def lineage(home, row):
    current = existing_rollout(home, row["rollout_path"])
    files, owners, seen = [], set(), set()
    cutoff = None
    while True:
        rid = rollout_id(current)
        if rid in seen:
            raise Error("会话前序记录存在循环。")
        seen.add(rid)
        data = rollout_bytes(current)
        meta = header(data, current)
        if not files and meta["id"] != row["id"]:
            raise Error("数据库所选记录不属于这个会话，拒绝迁移。")
        owners.add(meta["id"])
        if cutoff is not None:
            end = cutoff.get("end_byte_offset")
            if not isinstance(end, int) or end > len(data) or end < 1 or data[end-1:end] != b"\n":
                raise Error(f"前序记录边界无效：{current}")
        elif data and not data.endswith(b"\n"):
            raise Error(f"会话记录尚未完整写入：{current}；先退出对应客户端。")
        files.append((current, data, cutoff))
        base = meta.get("history_base") if meta.get("history_mode") == "paginated" else None
        if not base:
            break
        cutoff = base
        current = find_ancestor(home, base["thread_id"])
    return files, owners


def digest(home, row):
    files, _ = lineage(home, row)
    h = hashlib.sha256()
    # A fingerprint covers the selected rollout + frozen ancestor prefixes. Compression
    # and archive path changes do not look like new conversation turns.
    for _, data, cutoff in reversed(files):
        part = data[:cutoff["end_byte_offset"]] if cutoff else data
        head, separator, tail = part.partition(b"\n")
        normalized = json.loads(head)
        normalized["payload"].pop("model_provider", None)
        part = json.dumps(normalized, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode() + separator + tail
        h.update(len(part).to_bytes(8, "big"))
        h.update(part)
    return h.hexdigest()


@contextlib.contextmanager
def file_lock(path, message, shared=False):
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    f = path.open("a+b")
    try:
        try:
            fcntl.flock(f, (fcntl.LOCK_SH if shared else fcntl.LOCK_EX) | fcntl.LOCK_NB)
        except BlockingIOError:
            raise Error(message)
        yield
    finally:
        f.close()


@contextlib.contextmanager
def idle_homes(homes, tids):
    # Follow Codex's own lock order. Holding exclusive coordination also prevents
    # background compression and lock-file cleanup racing with snapshot/publication.
    with contextlib.ExitStack() as stack:
        for home in sorted(set(homes)):
            d = home / "thread-writer-locks"
            stack.enter_context(file_lock(d / ".coordination.lock", f"{home} 正在发布记录，请稍后重试。"))
            for tid in sorted(tids):
                stack.enter_context(file_lock(d / (tid + ".lock"), f"会话 {tid} 正在 {home.name} 中使用；先退出该会话再切换。"))
        yield


def table_snapshot(home, kind, names, tid, history_ids=None):
    p = db_path(home, kind)
    if not p:
        return {}
    out = {}
    with contextlib.closing(db_read(p)) as c:
        c.execute("BEGIN")
        available = tables(c)
        for name in names:
            if name not in available:
                continue
            key = "id" if name == "threads" else "thread_id"
            columns = [r[1] for r in c.execute(f'PRAGMA table_info("{name}")')]
            ids = history_ids if kind == "thread_history" and history_ids else [tid]
            rows = [dict(r) for r in c.execute(f'SELECT * FROM "{name}" WHERE "{key}" IN (' + ','.join('?' for _ in ids) + ')', ids)]
            out[name] = {"columns": columns, "rows": [{k: encode(v) for k, v in row.items()} for row in rows]}
    return out


def snapshot(home, tid, destination, allow_missing=False):
    row = thread_row(home, tid)
    if not row and not allow_missing:
        raise Error(f"{home} 中没有会话 {tid}。")
    destination.mkdir(parents=True, exist_ok=False, mode=0o700)
    manifest = {"format": 1, "cli_version": cli_version(), "thread_id": tid,
                "source_home": str(home), "exists": row is not None, "files": [], "databases": {}}
    if row:
        manifest["fingerprint"] = digest(home, row)
        files, _ = lineage(home, row)
        history_ids = {tid}
        for i, (path, data, cutoff) in enumerate(files):
            history_ids.add(header(data, path)["id"])
            if rollout_id(path):
                history_ids.add(rollout_id(path))
            rel = managed_path(home, path)
            target = destination / "files" / rel
            target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            shutil.copyfile(path, target)
            os.chmod(target, 0o600)
            manifest["files"].append({"path": str(rel), "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
                                       "primary": i == 0, "cutoff": cutoff})
        manifest["history_ids"] = sorted(history_ids)
        for kind, names in (("state", STATE_TABLES), ("thread_history", HISTORY_TABLES), ("goals", GOAL_TABLES)):
            manifest["databases"][kind] = table_snapshot(home, kind, names, tid, sorted(history_ids))
    write_json(destination / "manifest.json", manifest)
    return manifest


def cli_version():
    return subprocess.check_output([codex_binary(), "--version"], text=True, stderr=subprocess.DEVNULL).strip()


def codex_binary():
    p = shutil.which("codex")
    if not p:
        raise Error("找不到 codex 命令。")
    return p


def rpc_once(home, method, params):
    """One metadata-only request; never starts a model turn."""
    with tempfile.TemporaryDirectory(prefix="cswitch-init-") as tmp:
        env = dict(os.environ, CODEX_HOME=str(home))
        p = subprocess.Popen([codex_binary(), "app-server", "--stdio"], env=env, cwd=tmp,
                             stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
        try:
            msgs = [{"id": 1, "method": "initialize", "params": {"clientInfo": {"name": "cswitch", "version": VERSION}, "capabilities": {"experimentalApi": True}}},
                    {"method": "initialized"}, {"id": 2, "method": method, "params": params}]
            for msg in msgs:
                p.stdin.write(json.dumps(msg) + "\n")
            p.stdin.flush()
            import selectors
            sel = selectors.DefaultSelector()
            sel.register(p.stdout, selectors.EVENT_READ)
            import time
            deadline = time.monotonic() + 30
            while time.monotonic() < deadline:
                if not sel.select(timeout=1):
                    continue
                line = p.stdout.readline()
                if not line:
                    break
                msg = json.loads(line)
                if msg.get("id") == 2:
                    if "error" in msg:
                        raise Error("Codex 元数据请求失败：" + str(msg["error"]))
                    return msg["result"]
        finally:
            p.terminate()
            try:
                p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                p.kill()
                p.wait()
            p.stdin.close()
            p.stdout.close()
            if "sel" in locals():
                sel.close()
        raise Error(f"Codex 元数据请求超时或提前退出：{method}")


def ensure_database(home):
    if not db_path(home, "state"):
        rpc_once(home, "thread/list", {"limit": 1})
        if not db_path(home, "state"):
            raise Error(f"无法初始化 {home} 的数据库。")


def default_model(home):
    selected = config(home).get("model")
    if selected:
        return selected
    catalog = rpc_once(home, "model/list", {"limit": 100})
    for model in catalog.get("data", []):
        if model.get("isDefault"):
            return model["model"]
    raise Error("无法确定目标环境的默认模型。请在该环境的 config.toml 设置 model 后重试。")


def hydrate_history(home, tid, package):
    row = thread_row(home, tid)
    if row and row.get("history_mode") == "paginated" and not db_path(home, "thread_history"):
        # A fresh home has no history projection database yet. Ask Codex to create
        # and rebuild it from the durable lineage, so the UI shows old turns too.
        rpc_once(home, "thread/turns/list", {"threadId": tid, "limit": 1})
        # The paging request initializes the exact schema through Codex. Import
        # projections only after that initialization; paging alone doesn't rebuild
        # existing durable records in all CLI versions.
        apply_package(package, home, only_history=True)


def has_queue(home, tid):
    p = db_path(home, "queue")
    if not p:
        return False
    with contextlib.closing(db_read(p)) as c:
        return "queued_items" in tables(c) and c.execute("SELECT 1 FROM queued_items WHERE thread_id=? LIMIT 1", (tid,)).fetchone() is not None


def validate_package(package):
    m = read_json(package / "manifest.json")
    if not m or m.get("format") != 1:
        raise Error("不支持的会话包格式。")
    uuid.UUID(m["thread_id"])
    history_ids = m.get("history_ids", [m["thread_id"]])
    for value in history_ids:
        uuid.UUID(value)
    if m.get("exists") and (sum(bool(item.get("primary")) for item in m["files"]) != 1):
        raise Error("会话包必须包含唯一的主记录。")
    if len({item["path"] for item in m["files"]}) != len(m["files"]):
        raise Error("会话包包含重复文件路径。")
    for item in m["files"]:
        rel = Path(item["path"])
        if rel.is_absolute() or ".." in rel.parts or rel.parts[0] not in ("sessions", "archived_sessions"):
            raise Error("会话包含无效路径。")
        p = package / "files" / rel
        if not p.resolve().is_relative_to((package / "files").resolve()) or p.is_symlink() or not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest() != item["sha256"]:
            raise Error("会话包文件校验失败：" + str(rel))
    allowed = {"state": STATE_TABLES, "thread_history": HISTORY_TABLES, "goals": GOAL_TABLES}
    for kind, entries in m["databases"].items():
        if kind not in allowed or set(entries) - set(allowed[kind]):
            raise Error("会话包包含不支持的数据库表。")
        for name, table in entries.items():
            for column in table["columns"]:
                if not re.fullmatch(r"[a-z_][a-z_0-9]*", column):
                    raise Error("会话包包含无效数据库列名。")
            for row in table["rows"]:
                allowed_ids = history_ids if kind == "thread_history" else [m["thread_id"]]
                if set(row) != set(table["columns"]) or row.get("id" if name == "threads" else "thread_id") not in allowed_ids:
                    raise Error("会话包的数据库记录不属于所选会话。")
    return m


def plain_relative(item):
    p = Path(item["path"])
    return p.with_suffix("") if p.suffix == ".zst" else p


def apply_package(package, home, restore=False, only_history=False):
    """Publish only this thread. Never replace a whole database or copy credentials."""
    m = validate_package(package)
    tid = m["thread_id"]
    # Check immutable ancestors before writing anything. A conflicting ancestor may be
    # shared by other local threads; never overwrite it.
    for item in ([] if only_history else m["files"]):
        p = home / plain_relative(item)
        old_path = p if p.exists() else Path(str(p) + ".zst")
        if item["primary"] or not old_path.exists():
            continue
        old = rollout_bytes(old_path)
        new = rollout_bytes(package / "files" / item["path"])
        end = item["cutoff"]["end_byte_offset"]
        if old[:end] != new[:end]:
            raise Error("目标环境的前序记录发生分叉，拒绝覆盖：" + item["path"])
    # Resolve all column compatibility before mutation, including future versions.
    connections = []
    try:
        for kind, allowed in (("state", STATE_TABLES), ("thread_history", HISTORY_TABLES), ("goals", GOAL_TABLES)):
            if only_history and kind != "thread_history":
                continue
            entries = m["databases"].get(kind, {})
            p = db_path(home, kind)
            if not p:
                # thread_history is a rebuildable projection. Codex initializes its
                # versioned schema lazily from the copied durable rollout lineage.
                if kind != "thread_history" and any(v["rows"] for v in entries.values()):
                    raise Error(f"目标环境缺少 {kind} 数据库；请先运行一次该环境的 Codex。")
                continue
            c = sqlite3.connect(p, timeout=10)
            c.execute("PRAGMA foreign_keys=OFF")
            available = tables(c)
            connections.append((kind, c, available))
            for name, entry in entries.items():
                if name not in available:
                    if entry["rows"]:
                        raise Error(f"目标数据库缺少 {name} 表，版本不兼容。")
                    continue
                columns = {r[1] for r in c.execute(f'PRAGMA table_info("{name}")')}
                if set(entry["columns"]) - columns:
                    raise Error(f"{name} 数据库结构不同；请先把各环境更新到同一 Codex 版本。")
            c.execute("BEGIN IMMEDIATE")
        for item in reversed([] if only_history else m["files"]):
            p = home / plain_relative(item)
            if not item["primary"] and p.exists():
                continue
            p.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            fd, temp = tempfile.mkstemp(prefix=".cswitch-", dir=p.parent)
            os.close(fd)
            try:
                data = rollout_bytes(package / "files" / item["path"])
                meta = header(data, p)
                if item["primary"] and meta.get("history_mode", "legacy") == "legacy" and not restore:
                    # Legacy discovery repairs SQLite from the JSONL header before
                    # validating the provider; update that header too. Never change
                    # paginated headers, whose byte offsets may be inherited.
                    head, sep, tail = data.partition(b"\n")
                    value = json.loads(head)
                    value["payload"]["model_provider"] = config(home).get("model_provider", "openai")
                    data = json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode() + sep + tail
                Path(temp).write_bytes(data)
                os.chmod(temp, 0o600)
                os.replace(temp, p)
            finally:
                if os.path.exists(temp):
                    os.unlink(temp)
        for kind, c, available in connections:
            names = {"state": STATE_TABLES, "thread_history": HISTORY_TABLES, "goals": GOAL_TABLES}[kind]
            entries = m["databases"].get(kind, {})
            for name in reversed(names):
                if name in available:
                    key = "id" if name == "threads" else "thread_id"
                    c.execute(f'DELETE FROM "{name}" WHERE "{key}"=?', (tid,))
            for name in names:
                entry = entries.get(name)
                if not entry or name not in available:
                    continue
                for source_row in entry["rows"]:
                    row = {k: decode(v) for k, v in source_row.items()}
                    if name == "threads":
                        source_home = Path(m["source_home"])
                        rel = managed_path(source_home, row["rollout_path"])
                        if rel.suffix == ".zst":
                            rel = rel.with_suffix("")
                        # State paths may intentionally omit .zst; Codex supports that lookup.
                        row["rollout_path"] = str(home / rel)
                        if not restore:
                            # Codex validates the persisted provider before applying
                            # resume overrides. Rebind metadata to the destination;
                            # original historical rollout bytes remain unchanged.
                            destination_config = config(home)
                            row["model_provider"] = destination_config.get("model_provider", "openai")
                            if "model" in row:
                                row["model"] = destination_config.get("model")
                            # Organization and remote ownership remain local to each environment.
                            for k in ("project_id", "thread_section_id", "section_position", "section_entered_at_ms", "attached_thread_parent_id"):
                                if k in row:
                                    row[k] = None
                            row["archived"], row["archived_at"] = 0, None
                    cols = list(row)
                    verb = "INSERT OR IGNORE" if kind == "thread_history" and row.get("thread_id") != tid else "INSERT"
                    c.execute(f'{verb} INTO "{name}" (' + ",".join('"' + k + '"' for k in cols) + ') VALUES (' + ",".join("?" for _ in cols) + ')', [row[k] for k in cols])
        for _, c, _ in connections:
            c.commit()
    except BaseException:
        for _, c, _ in connections:
            c.rollback()
        raise
    finally:
        for _, c, _ in connections:
            c.close()
    return m


class Switcher:
    def __init__(self, settings=None):
        self.root = Path(os.environ.get("CSWITCH_HOME", Path.home() / ".local/share/cswitch")).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.settings_path = Path(settings).expanduser() if settings else self.root / "config.json"
        default = {"homes": {"a": str(Path.home()/".codex"), "b": str(Path.home()/".codex-b"), "c": str(Path.home()/".codex-c")}}
        if not self.settings_path.exists():
            write_json(self.settings_path, default)
        settings = read_json(self.settings_path)
        self.homes = {k: Path(v).expanduser().resolve() for k, v in settings["homes"].items()}
        if len(set(self.homes.values())) != len(self.homes):
            raise Error("多个环境指向同一个目录。")
        self.registry = read_json(self.root / "registry.json", {})

    def name(self, value):
        aliases = {"codex": "a", "codex-b": "b", "cb": "b", "codex-c": "c", "cnew": "c"}
        key = aliases.get(value, value)
        if key not in self.homes or not self.homes[key].is_dir():
            raise Error("未知环境：" + value + "；可用环境：" + ", ".join(self.homes))
        return key

    def candidates(self, tid):
        result = {}
        for key, home in self.homes.items():
            row = thread_row(home, tid)
            if row:
                result[key] = row
        if not result:
            raise Error("找不到会话 " + tid + "。用 cswitch list 查看会话。")
        return result

    def source(self, tid, candidates, explicit=None):
        if explicit:
            key = self.name(explicit)
            if key not in candidates:
                raise Error(f"{key} 中没有这个会话。")
            return key
        if len(candidates) == 1:
            return next(iter(candidates))
        record = self.registry.get(tid, {})
        baseline = record.get("copies", {})
        fingerprints = {key: digest(self.homes[key], row) for key, row in candidates.items()}
        changed = [key for key, fingerprint in fingerprints.items() if fingerprint != baseline.get(key)]
        if len(changed) == 1:
            return changed[0]
        if changed and len({fingerprints[key] for key in changed}) == 1:
            return record.get("owner") if record.get("owner") in changed else changed[0]
        if len(set(fingerprints.values())) == 1:
            return record.get("owner") if record.get("owner") in candidates else next(iter(candidates))
        if not changed and record.get("owner") in candidates:
            return record["owner"]
        raise Error("多个环境中的同名会话存在不同进度，无法自动判断。用 --from a/b/c 选择来源；目标也有独立修改时还需 --replace（会先备份）。")

    def choose(self, selector, all_cwds=False, source=None, limit=25):
        if selector:
            try:
                return str(uuid.UUID(selector))
            except ValueError:
                # Unique UUID prefix is convenient without fuzzy-matching private titles.
                ids = {r["id"] for h in self.homes.values() for r in list_rows(h) if r["id"].startswith(selector)}
                if len(ids) == 1:
                    return ids.pop()
                raise Error("会话 ID 无效或前缀不唯一。")
        items = self.listing(all_cwds, source, limit)
        if not items:
            raise Error("当前目录没有会话。使用 --all 查看所有目录。")
        self.print_listing(items)
        if not sys.stdin.isatty():
            raise Error("非交互调用请提供会话 ID。")
        value = input("选择序号（回车取消）：").strip()
        if not value:
            raise Error("已取消。")
        if not value.isdigit() or not 1 <= int(value) <= len(items):
            raise Error("无效序号。")
        return items[int(value)-1]["id"]

    def listing(self, all_cwds=False, source=None, limit=25):
        grouped = {}
        keys = [self.name(source)] if source else self.homes
        for key in keys:
            for row in list_rows(self.homes[key]):
                if not all_cwds and Path(row["cwd"]).resolve() != Path.cwd().resolve():
                    continue
                if row.get("agent_role") or row.get("attached_thread_parent_id"):
                    continue
                tid = row["id"]
                if tid not in grouped:
                    grouped[tid] = dict(row, environments=[])
                grouped[tid]["environments"].append(key)
                if row["updated_at"] > grouped[tid]["updated_at"]:
                    envs = grouped[tid]["environments"]
                    grouped[tid] = dict(row, environments=envs)
        return sorted(grouped.values(), key=lambda r: r["updated_at"], reverse=True)[:limit]

    @staticmethod
    def print_listing(items):
        for i, row in enumerate(items, 1):
            title = " ".join(re.sub(r"[\x00-\x1f\x7f-\x9f]", "", row.get("name") or row.get("title") or "无标题").split())[:65]
            stamp = dt.datetime.fromtimestamp(row["updated_at"]).strftime("%m-%d %H:%M")
            print(f'{i:2}. [{",".join(row["environments"])}] {stamp}  {title}')
            print(f'    {row["id"]}  {row["cwd"]}')

    def transfer(self, target, tid, source=None, replace=False, no_launch=False, dry_run=False):
        target = self.name(target)
        if not (no_launch or dry_run) and not sys.stdin.isatty():
            raise Error("启动交互会话需要终端；仅同步请加 --no-launch。")
        launch_model = default_model(self.homes[target]) if not (no_launch or dry_run) else None
        with file_lock(self.root / "locks" / (tid + ".lock"), "该会话已有 cswitch 正在操作或运行。"):
            self.registry = read_json(self.root / "registry.json", {})
            pending = []
            for journal_path in (self.root / "backups" / tid).glob("*/transaction.json"):
                if read_json(journal_path).get("status") == "prepared":
                    pending.append(str(journal_path.parent))
            if pending:
                raise Error("发现上次未完成的切换；为防止覆盖已恢复的工作，已停止。备份位置：" + ", ".join(pending))
            candidates = self.candidates(tid)
            # Briefly freeze all configured homes so source selection cannot race a turn.
            _, owners = lineage(self.homes[next(iter(candidates))], next(iter(candidates.values())))
            for key, row in candidates.items():
                _, ids = lineage(self.homes[key], row)
                owners.update(ids)
            homes = [self.homes[key] for key in set(candidates) | {target}]
            if not dry_run:
                ensure_database(self.homes[target])
            with idle_homes(homes, owners | {tid}):
                candidates = self.candidates(tid)
                selected = self.source(tid, candidates, source)
                src, dst = self.homes[selected], self.homes[target]
                row = candidates[selected]
                fp = digest(src, row)
                print(f'会话：{tid}\n来源：{selected} ({src})\n目标：{target} ({dst})\n项目：{row["cwd"]}', flush=True)
                if not Path(row["cwd"]).is_dir():
                    raise Error("项目目录不存在，不能恢复。")
                if has_queue(src, tid) or has_queue(dst, tid):
                    raise Error("该会话还有排队消息。请先处理或清除，再切换。")
                if target in candidates and selected != target:
                    target_fp = digest(dst, candidates[target])
                    known = self.registry.get(tid, {}).get("copies", {}).get(target)
                    if target_fp != fp and target_fp != known and not replace:
                        raise Error("目标会话有未同步的独立修改，拒绝覆盖。明确选择 --from 来源并加 --replace 才会备份后覆盖。")
                if dry_run:
                    print("检查通过；未复制会话，也未启动 Codex。")
                    return 0
                if selected != target:
                    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S-%f")
                    work = self.root / "backups" / tid / stamp
                    work.mkdir(parents=True, mode=0o700)
                    snapshot(src, tid, work / "incoming")
                    snapshot(dst, tid, work / "previous", allow_missing=True)
                    incoming = validate_package(work / "incoming")
                    newly_created = [dst / plain_relative(item) for item in incoming["files"] if not (dst / plain_relative(item)).exists()]
                    journal = {"status": "prepared", "target": target, "thread_id": tid, "backup": str(work)}
                    write_json(work / "transaction.json", journal)
                    try:
                        apply_package(work / "incoming", dst)
                        hydrate_history(dst, tid, work / "incoming")
                        if digest(dst, thread_row(dst, tid)) != fp:
                            raise Error("复制后的完整历史校验失败。")
                    except BaseException:
                        apply_package(work / "previous", dst, restore=True)
                        for p in newly_created:
                            p.unlink(missing_ok=True)
                        journal["status"] = "rolled_back"
                        write_json(work / "transaction.json", journal)
                        raise
                    journal["status"] = "complete"
                    write_json(work / "transaction.json", journal)
                    print("同步完成；备份：" + str(work), flush=True)
                copies = {key: digest(self.homes[key], candidate) for key, candidate in candidates.items()}
                copies[target] = fp
                self.registry[tid] = {"owner": target, "copies": copies}
                # Registry writes from different session processes must merge, not overwrite.
                with file_lock(self.root / "registry.lock", "另一个切换正在更新索引，请重试。"):
                    latest = read_json(self.root / "registry.json", {})
                    latest[tid] = self.registry[tid]
                    write_json(self.root / "registry.json", latest)
            if no_launch:
                print(f"已就绪。继续工作：cswitch {target} {tid}")
                return 0
            # Explicitly select the destination provider/model rather than inheriting the
            # source thread's most recent model. Config/auth stay entirely in destination.
            conf = config(self.homes[target])
            provider = conf.get("model_provider", "openai")
            command = [codex_binary(), "--no-daemon", "-c", "model_provider=" + json.dumps(provider)]
            command += ["-m", launch_model]
            command += ["resume", tid]
            print(f"启动 {target}（provider={provider}, model={launch_model}）", flush=True)
            child = subprocess.Popen(command, env=dict(os.environ, CODEX_HOME=str(self.homes[target])), cwd=row["cwd"])
            while True:
                try:
                    return child.wait()
                except KeyboardInterrupt:
                    # The foreground terminal also delivers SIGINT to Codex. Keep the
                    # session lock held until Codex actually exits.
                    continue


def parser():
    p = argparse.ArgumentParser(description="在独立 Codex 环境间接力同一个 session；a=原版，b=codex-b，c=codex-c。")
    p.add_argument("target", nargs="?", help="目标环境 a/b/c；或 list/export/import")
    p.add_argument("session", nargs="?", help="会话 UUID 或唯一前缀；不填则交互选择")
    p.add_argument("path", nargs="?", help="export/import 的会话包目录")
    p.add_argument("--from", dest="source", help="明确指定来源环境")
    p.add_argument("--all", action="store_true", help="选择器列出所有项目的会话")
    p.add_argument("--limit", type=int, default=25, help="列表最多显示多少个会话（默认 25）")
    p.add_argument("--no-launch", action="store_true", help="只同步，不启动 Codex")
    p.add_argument("--dry-run", action="store_true", help="检查来源、冲突和锁，不复制/启动")
    p.add_argument("--replace", action="store_true", help="明确允许备份后覆盖目标的分叉进度")
    p.add_argument("--settings", help="自定义环境映射 JSON 文件")
    p.add_argument("--version", action="version", version=VERSION)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    if not args.target:
        parser().print_help()
        return 0
    s = Switcher(args.settings)
    if args.target == "list":
        s.print_listing(s.listing(args.all, args.source, args.limit))
        return 0
    if args.target == "export":
        if not args.session or not args.path:
            raise Error("用法：cswitch export 会话ID 会话包目录 [--from a/b/c]")
        tid = s.choose(args.session)
        with file_lock(s.root / "locks" / (tid + ".lock"), "该会话已有 cswitch 正在操作。"):
            candidates = s.candidates(tid)
            with idle_homes([s.homes[k] for k in candidates], {tid}):
                selected = s.source(tid, candidates, args.source)
                files, owners = lineage(s.homes[selected], candidates[selected])
                # Coordination remains held, preventing new writers/compression; validate
                # every ancestor owner too, not just the requested root.
                with contextlib.ExitStack() as locks:
                    for owner in sorted(owners - {tid}):
                        locks.enter_context(file_lock(s.homes[selected]/"thread-writer-locks"/(owner+".lock"), "前序会话仍在运行，请先退出。"))
                    snapshot(s.homes[selected], tid, Path(args.path).expanduser().resolve())
        print("已导出：" + str(Path(args.path).expanduser().resolve()))
        return 0
    if args.target == "import":
        if not args.session or not args.path:
            raise Error("用法：cswitch import 目标环境 会话包目录 [--replace]")
        # Import is deliberately separate from launching; a portable package is never
        # allowed to choose an executable, environment mapping, or a model API.
        target = s.name(args.session)
        package = Path(args.path).expanduser().resolve()
        m = validate_package(package)
        if not m.get("exists"):
            raise Error("这个备份表示目标原先不存在该会话，不能作为普通会话包导入。")
        tid = m["thread_id"]
        ensure_database(s.homes[target])
        with file_lock(s.root/"locks"/(tid+".lock"), "该会话正在使用。"), idle_homes([s.homes[target]], {tid}):
            existing = thread_row(s.homes[target], tid)
            if existing and not args.replace:
                raise Error("目标已有同 ID 会话；加 --replace 可备份后覆盖。")
            stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S-%f")
            backup = s.root/"backups"/tid/(stamp+"-import")
            snapshot(s.homes[target], tid, backup, allow_missing=True)
            newly_created = [s.homes[target] / plain_relative(item) for item in m["files"] if not (s.homes[target] / plain_relative(item)).exists()]
            try:
                apply_package(package, s.homes[target])
                hydrate_history(s.homes[target], tid, package)
                imported = thread_row(s.homes[target], tid)
                if not imported or digest(s.homes[target], imported) != m["fingerprint"]:
                    raise Error("导入后的完整历史校验失败。")
            except BaseException:
                apply_package(backup, s.homes[target], restore=True)
                for p in newly_created:
                    p.unlink(missing_ok=True)
                raise
            with file_lock(s.root/"registry.lock", "另一个切换正在更新索引，请重试。"):
                registry = read_json(s.root/"registry.json", {})
                record = registry.get(tid, {"copies": {}})
                record["owner"] = target
                for key, candidate in s.candidates(tid).items():
                    fp = digest(s.homes[key], candidate)
                    if fp == m["fingerprint"]:
                        record["copies"][key] = fp
                registry[tid] = record
                write_json(s.root/"registry.json", registry)
        print(f"已导入 {target}；继续工作：cswitch {target} {tid}\n备份：{backup}")
        return 0
    if args.path:
        raise Error("多余参数：" + args.path)
    tid = s.choose(args.session, args.all, args.source, args.limit)
    return s.transfer(args.target, tid, args.source, args.replace, args.no_launch, args.dry_run)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (Error, ValueError, OSError, sqlite3.Error) as e:
        print("cswitch: " + str(e), file=sys.stderr)
        raise SystemExit(2)
    except KeyboardInterrupt:
        raise SystemExit(130)
