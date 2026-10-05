from __future__ import annotations

import logging
from pathlib import Path

from sqlalchemy import event, inspect
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker, create_async_engine

from lalascaner.db.models import Base

log = logging.getLogger(__name__)


def make_engine(url: str) -> AsyncEngine:
    if url.startswith("sqlite") and ":///" in url:
        db_path = url.split(":///", 1)[1]
        if db_path and db_path != ":memory:":
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    is_sqlite = url.startswith("sqlite")
    engine = create_async_engine(url, connect_args={"timeout": 60} if is_sqlite else {})
    if is_sqlite:

        @event.listens_for(engine.sync_engine, "connect")
        def _sqlite_pragmas(dbapi_conn, _record):  # noqa: ANN001
            cur = dbapi_conn.cursor()
            cur.execute("PRAGMA journal_mode=WAL")
            cur.execute("PRAGMA synchronous=NORMAL")
            cur.execute("PRAGMA foreign_keys=ON")
            cur.execute("PRAGMA busy_timeout=60000")
            cur.close()

    return engine


def _add_missing_columns(sync_conn) -> None:  # noqa: ANN001
    """Простейшая миграция: добавляет в существующие таблицы колонки, появившиеся в моделях."""
    insp = inspect(sync_conn)
    for table in Base.metadata.sorted_tables:
        if not insp.has_table(table.name):
            continue
        existing = {col["name"] for col in insp.get_columns(table.name)}
        for col in table.columns:
            if col.name in existing:
                continue
            ddl_type = col.type.compile(dialect=sync_conn.dialect)
            default_sql = ""
            if col.default is not None and not callable(col.default.arg):
                value = col.default.arg
                if isinstance(value, bool):
                    default_sql = f" DEFAULT {int(value)}"
                elif isinstance(value, (int, float)):
                    default_sql = f" DEFAULT {value}"
                elif isinstance(value, str):
                    default_sql = " DEFAULT '{}'".format(value.replace("'", "''"))
            sync_conn.exec_driver_sql(f"ALTER TABLE {table.name} ADD COLUMN {col.name} {ddl_type}{default_sql}")
            log.info("миграция: добавлена колонка %s.%s", table.name, col.name)
        for index in table.indexes:
            index.create(bind=sync_conn, checkfirst=True)


async def init_db(engine: AsyncEngine) -> None:
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        await conn.run_sync(_add_missing_columns)


def make_sessionmaker(engine: AsyncEngine) -> async_sessionmaker:
    return async_sessionmaker(engine, expire_on_commit=False)
