"""Кнопка «Обновить с GitHub»: git fetch + fast-forward прямо из бота, без захода на сервер руками.

Работает, когда каталог проекта смонтирован в контейнер как есть (docker-compose.yml монтирует весь
репозиторий в /app, а не только собранный на этапе сборки код) и пакет установлен editable-режимом
(`pip install -e .`) — тогда новые файлы из git сразу видны Python, и после перезапуска процесса
подтягивается свежий код без пересборки образа. Если поменялись зависимости (pyproject.toml) — только
пересборка образа их поставит, поэтому в этом случае обновление файлов делаем, но процесс не трогаем.
"""

from __future__ import annotations

import asyncio
import os
import sys
from dataclasses import dataclass, field

GIT_TIMEOUT = 60


class GitError(Exception):
    """git недоступен, не настроен upstream, есть локальные изменения и т.п. — сообщаем текст пользователю."""


@dataclass
class UpdateResult:
    updated: bool
    commits: list[str] = field(default_factory=list)
    needs_rebuild: bool = False
    head: str = ""


async def _git(repo_path: str, *args: str) -> str:
    try:
        proc = await asyncio.create_subprocess_exec(
            "git", "-C", repo_path, *args,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=GIT_TIMEOUT)
    except FileNotFoundError as exc:
        raise GitError("git не найден в контейнере") from exc
    except TimeoutError as exc:
        raise GitError(f"git {' '.join(args)} не ответил за {GIT_TIMEOUT} с (нет сети?)") from exc
    if proc.returncode != 0:
        raise GitError(stderr.decode("utf-8", "replace").strip() or f"git {args[0]} завершился с ошибкой")
    return stdout.decode("utf-8", "replace").strip()


async def check_for_updates(repo_path: str) -> UpdateResult:
    """git fetch + сравнение с upstream, без изменения рабочей копии."""
    await _git(repo_path, "fetch", "--quiet")
    ahead = await _git(repo_path, "rev-list", "HEAD..@{upstream}", "--count")
    if int(ahead) == 0:
        head = await _git(repo_path, "rev-parse", "--short", "HEAD")
        return UpdateResult(updated=False, head=head)
    log = await _git(repo_path, "log", "--oneline", "--no-decorate", "HEAD..@{upstream}")
    changed = await _git(repo_path, "diff", "--name-only", "HEAD..@{upstream}")
    needs_rebuild = any(name in changed.splitlines() for name in ("pyproject.toml", "Dockerfile"))
    return UpdateResult(updated=True, commits=log.splitlines(), needs_rebuild=needs_rebuild)


async def apply_update(repo_path: str) -> UpdateResult:
    """check_for_updates + сам git merge --ff-only, если позади. Не трогает рабочую копию, если уже свежая."""
    result = await check_for_updates(repo_path)
    if not result.updated:
        return result
    await _git(repo_path, "merge", "--ff-only", "@{upstream}")
    result.head = await _git(repo_path, "rev-parse", "--short", "HEAD")
    return result


def restart_process() -> None:
    """Заменяет текущий процесс собой же — новый интерпретатор перечитает уже обновлённые (editable-install)
    модули с диска. Работает и в Docker (сохраняет PID 1, не нужен restart: unless-stopped), и без него."""
    os.execv(sys.executable, [sys.executable, "-m", "lalascaner", "run"])
