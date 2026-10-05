"""Команда /backup: выгружает текущую SQLite-базу файлом прямо в чат.

Снимок делается через sqlite3.backup(), а не копированием .db-файла с диска — база работает
в WAL-режиме (см. db/session.py), и свежие записи могут лежать в .db-wal, не попав в простую
копию до checkpoint'а. backup() даёт консистентный снимок на момент вызова вне зависимости от этого.
"""

from __future__ import annotations

import math
import os
import sqlite3
import tempfile
from pathlib import Path

PART_SIZE = 25 * 1024 * 1024
DERIVED_TABLE = "listing_vectors"


class NotSqliteError(Exception):
    """DATABASE_URL указывает не на SQLite — бэкап файлом не имеет смысла."""


def sqlite_path(database_url: str) -> str:
    if not database_url.startswith("sqlite") or ":///" not in database_url:
        raise NotSqliteError(database_url)
    return database_url.split(":///", 1)[1]


def make_backup(database_url: str) -> Path:
    """Блокирующая функция — вызывать через asyncio.to_thread. Возвращает путь к временному файлу,
    удалить который — забота вызывающего кода."""
    src_path = sqlite_path(database_url)
    fd, dest_path = tempfile.mkstemp(suffix=".db", prefix="lalascaner-backup-")
    os.close(fd)
    src_conn = sqlite3.connect(src_path)
    try:
        dest_conn = sqlite3.connect(dest_path)
        try:
            src_conn.backup(dest_conn)
            # Векторы — производные данные (пересчитываются сами) и самая тяжёлая таблица: в бэкап не берём.
            dest_conn.execute(f"DROP TABLE IF EXISTS {DERIVED_TABLE}")
            dest_conn.execute("VACUUM")
        finally:
            dest_conn.close()
    finally:
        src_conn.close()
    return Path(dest_path)


def split_file(path: Path, part_size: int = PART_SIZE) -> list[Path]:
    """Режет файл на куски не больше part_size рядом с ним (file.part01, file.part02…). Файл, который
    и так влезает, возвращается как есть. Склеиваются обратно простой конкатенацией по порядку имён."""
    size = path.stat().st_size
    if size <= part_size:
        return [path]
    count = math.ceil(size / part_size)
    width = len(str(count))
    parts = []
    with path.open("rb") as src:
        for i in range(1, count + 1):
            part = path.with_name(f"{path.name}.part{i:0{width}d}")
            part.write_bytes(src.read(part_size))
            parts.append(part)
    return parts
