import pathlib
import pytest

REPO_ROOT = pathlib.Path(__file__).parent.parent

@pytest.fixture
def repo_root() -> pathlib.Path:
    return REPO_ROOT

@pytest.fixture
def fixtures_dir(repo_root) -> pathlib.Path:
    return repo_root / "tests" / "fixtures"
