from pathlib import Path

import pytest

from lalascaner.db.session import init_db, make_engine, make_sessionmaker

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def fixtures() -> Path:
    return FIXTURES


@pytest.fixture
async def maker(tmp_path):
    engine = make_engine(f"sqlite+aiosqlite:///{tmp_path / 'p.db'}")
    await init_db(engine)
    yield make_sessionmaker(engine)
    await engine.dispose()
