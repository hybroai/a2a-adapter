"""Checks that the release-please setup covers every place the version lives.

release-please bumps the version in its manifest, pyproject.toml and
a2a_adapter/__init__.py (python strategy); the release-please workflow then
runs ``uv lock`` on the release PR. If these drift apart, the next release PR
would publish a mismatched package, and ``uv sync --locked`` would fail.
"""

import json
import tomllib
from pathlib import Path

import a2a_adapter

ROOT = Path(__file__).resolve().parents[2]


def _pyproject_version() -> str:
    with open(ROOT / "pyproject.toml", "rb") as f:
        return tomllib.load(f)["project"]["version"]


def _lock_version() -> str:
    with open(ROOT / "uv.lock", "rb") as f:
        packages = tomllib.load(f)["package"]
    return next(p["version"] for p in packages if p["name"] == "a2a-adapter")


def test_manifest_matches_package_version():
    manifest = json.loads((ROOT / ".release-please-manifest.json").read_text())
    assert manifest == {".": _pyproject_version()}


def test_all_version_locations_agree():
    assert a2a_adapter.__version__ == _pyproject_version() == _lock_version()


def test_release_please_config_and_lock_sync():
    config = json.loads((ROOT / "release-please-config.json").read_text())
    package = config["packages"]["."]
    # The python strategy updates pyproject.toml and a2a_adapter/__init__.py.
    assert package["release-type"] == "python"
    # release-please's TOML updater has not updated uv.lock in practice, so the workflow
    # relocks the release PR instead.
    workflow = (ROOT / ".github/workflows/release-please.yml").read_text()
    assert "sync-lock:" in workflow
    assert '--label "autorelease: pending"' in workflow
    assert "uv lock\n" in workflow
    # publish.yml checks the tag against v<version>.
    assert config["include-component-in-tag"] is False
    assert config.get("include-v-in-tag", True) is True
