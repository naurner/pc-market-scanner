"""/backup: снимок SQLite-базы через sqlite3.backup(), а не копирование файла с диска —
важно, что он видит и данные, ещё не сброшенные из WAL (см. комментарий в bot/backup.py)."""

import sqlite3

import pytest

from lalascaner.bot.backup import NotSqliteError, make_backup, split_file, sqlite_path


def test_sqlite_path_extracts_filesystem_path():
    assert sqlite_path("sqlite+aiosqlite:///data/lalascaner.db") == "data/lalascaner.db"
    assert sqlite_path("sqlite:////abs/path.db") == "/abs/path.db"


def test_sqlite_path_rejects_non_sqlite_url():
    with pytest.raises(NotSqliteError):
        sqlite_path("postgresql+asyncpg://user:pass@host/db")


def test_make_backup_captures_wal_only_data(tmp_path):
    db_path = tmp_path / "lalascaner.db"
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, v TEXT)")
        conn.execute("INSERT INTO t (v) VALUES ('checkpointed')")
        conn.commit()
        conn.execute("PRAGMA wal_checkpoint(FULL)")
        conn.execute("INSERT INTO t (v) VALUES ('только в wal')")
        conn.commit()  # не чекпоинтится — остаётся в -wal файле

        backup_path = make_backup(f"sqlite+aiosqlite:///{db_path}")
    finally:
        conn.close()

    try:
        backup_conn = sqlite3.connect(backup_path)
        try:
            rows = backup_conn.execute("SELECT v FROM t ORDER BY id").fetchall()
        finally:
            backup_conn.close()
        assert [r[0] for r in rows] == ["checkpointed", "только в wal"]
    finally:
        backup_path.unlink()


def test_make_backup_is_independent_copy(tmp_path):
    db_path = tmp_path / "lalascaner.db"
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("CREATE TABLE t (v TEXT)")
        conn.commit()

        backup_path = make_backup(f"sqlite+aiosqlite:///{db_path}")
        conn.execute("INSERT INTO t VALUES ('после бэкапа')")
        conn.commit()
    finally:
        conn.close()

    try:
        backup_conn = sqlite3.connect(backup_path)
        try:
            rows = backup_conn.execute("SELECT * FROM t").fetchall()
        finally:
            backup_conn.close()
        assert rows == []  # изменения после снимка не попадают в уже сделанный файл
    finally:
        backup_path.unlink()


def test_split_file_small_file_is_returned_as_is(tmp_path):
    path = tmp_path / "db.db"
    path.write_bytes(b"x" * 100)
    assert split_file(path, part_size=100) == [path]


def test_split_file_parts_concatenate_back_to_original(tmp_path):
    path = tmp_path / "db.db"
    data = bytes(range(256)) * 50  # 12 800 байт
    path.write_bytes(data)
    parts = split_file(path, part_size=1000)
    assert len(parts) == 13 and all(p.stat().st_size <= 1000 for p in parts)
    assert [p.name for p in parts] == sorted(p.name for p in parts)  # `cat *.part*` склеит в нужном порядке
    assert parts[0].name == "db.db.part01" and parts[-1].name == "db.db.part13"
    assert b"".join(p.read_bytes() for p in parts) == data


def test_split_file_exact_multiple_has_no_empty_tail(tmp_path):
    path = tmp_path / "db.db"
    path.write_bytes(b"y" * 3000)
    parts = split_file(path, part_size=1000)
    assert [p.stat().st_size for p in parts] == [1000, 1000, 1000]


def test_make_backup_drops_derived_vectors_table(tmp_path):
    db_path = tmp_path / "lalascaner.db"
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("CREATE TABLE listings (id INTEGER PRIMARY KEY)")
        conn.execute("CREATE TABLE listing_vectors (listing_id INTEGER PRIMARY KEY, vec BLOB)")
        conn.execute("INSERT INTO listing_vectors VALUES (1, randomblob(2048))")
        conn.commit()
    finally:
        conn.close()
    backup_path = make_backup(f"sqlite+aiosqlite:///{db_path}")
    try:
        backup_conn = sqlite3.connect(backup_path)
        try:
            tables = {r[0] for r in backup_conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        finally:
            backup_conn.close()
        assert tables == {"listings"}
    finally:
        backup_path.unlink()
