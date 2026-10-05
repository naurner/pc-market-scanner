"""Кнопка «Обновить с GitHub»: git fetch + fast-forward. Проверяется на настоящих git-репозиториях
во временных папках — без сети, но с реальным git, а не имитацией его вывода."""

import asyncio
import subprocess
import tempfile
from pathlib import Path

import pytest

from lalascaner.bot.update import GitError, apply_update, check_for_updates


def git(repo: str, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", repo, *args], check=True, capture_output=True, text=True, encoding="utf-8"
    ).stdout.strip()


def init_repo_pair(tmp_path) -> tuple[str, str]:
    """(origin, clone) — «origin» голый репозиторий, «clone» — рабочая копия с настроенным upstream,
    как после `git clone` на сервере пользователя."""
    origin = str(tmp_path / "origin.git")
    clone = str(tmp_path / "clone")
    subprocess.run(["git", "init", "--quiet", "--bare", origin], check=True)
    subprocess.run(["git", "clone", "--quiet", origin, clone], check=True)
    for key, value in (("user.email", "bot@example.com"), ("user.name", "bot")):
        git(clone, "config", key, value)
    (tmp_path / "clone" / "README.md").write_text("v1\n", encoding="utf-8")
    git(clone, "add", ".")
    git(clone, "commit", "--quiet", "-m", "начальный коммит")
    git(clone, "push", "--quiet", "origin", "HEAD")
    return origin, clone


def push_commit(origin: str, message: str, filename: str = "README.md", content: str = "v2\n") -> None:
    """Коммитит и пушит в origin из отдельного клона — как будто кто-то другой запушил обновление на GitHub."""
    with tempfile.TemporaryDirectory() as other:
        subprocess.run(["git", "clone", "--quiet", origin, other], check=True)
        git(other, "config", "user.email", "dev@example.com")
        git(other, "config", "user.name", "dev")
        (Path(other) / filename).write_text(content, encoding="utf-8")
        git(other, "add", ".")
        git(other, "commit", "--quiet", "-m", message)
        git(other, "push", "--quiet", "origin", "HEAD")


async def test_up_to_date_reports_head_without_changing_anything(tmp_path):
    _, clone = init_repo_pair(tmp_path)
    before = git(clone, "rev-parse", "HEAD")
    result = await check_for_updates(clone)
    assert not result.updated
    assert before.startswith(result.head) and result.head
    assert git(clone, "rev-parse", "HEAD") == before  # рабочая копия не тронута


async def test_check_reports_pending_commits_without_merging(tmp_path):
    origin, clone = init_repo_pair(tmp_path)
    before = git(clone, "rev-parse", "HEAD")
    push_commit(origin, "добавили фичу")
    result = await check_for_updates(clone)
    assert result.updated and not result.needs_rebuild
    assert any("добавили фичу" in line for line in result.commits)
    assert git(clone, "rev-parse", "HEAD") == before  # check_for_updates не мержит


async def test_apply_update_fast_forwards_and_returns_new_head(tmp_path):
    origin, clone = init_repo_pair(tmp_path)
    push_commit(origin, "фикс бага")
    result = await apply_update(clone)
    assert result.updated
    assert (Path(clone) / "README.md").read_text(encoding="utf-8") == "v2\n"
    assert git(clone, "rev-parse", "HEAD").startswith(result.head)


async def test_pyproject_change_is_flagged_for_rebuild(tmp_path):
    origin, clone = init_repo_pair(tmp_path)
    push_commit(origin, "новая зависимость", filename="pyproject.toml", content="[project]\nname='x'\n")
    result = await apply_update(clone)
    assert result.updated and result.needs_rebuild
    assert (Path(clone) / "pyproject.toml").exists()  # файлы всё равно подтянулись


async def test_second_apply_is_a_noop(tmp_path):
    origin, clone = init_repo_pair(tmp_path)
    push_commit(origin, "коммит")
    first = await apply_update(clone)
    second = await apply_update(clone)
    assert first.updated and not second.updated
    assert second.head == first.head


async def test_no_upstream_raises_git_error(tmp_path):
    repo = str(tmp_path / "solo")
    subprocess.run(["git", "init", "--quiet", repo], check=True)
    git(repo, "config", "user.email", "a@b.c")
    git(repo, "config", "user.name", "a")
    (tmp_path / "solo" / "f.txt").write_text("x", encoding="utf-8")
    git(repo, "add", ".")
    git(repo, "commit", "--quiet", "-m", "x")
    with pytest.raises(GitError):
        await check_for_updates(repo)


async def test_missing_repo_raises_git_error(tmp_path):
    with pytest.raises(GitError):
        await check_for_updates(str(tmp_path / "does-not-exist"))


async def test_git_binary_missing_raises_git_error(monkeypatch, tmp_path):
    async def boom(*a, **k):
        raise FileNotFoundError("no git")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", boom)
    with pytest.raises(GitError, match="git не найден"):
        await check_for_updates(str(tmp_path))
