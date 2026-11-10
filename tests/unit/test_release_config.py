"""Checks that the release-please setup covers every place the version lives.

release-please bumps the version in its manifest, pyproject.toml and
a2a_adapter/__init__.py (python strategy); the release-please workflow then
runs ``uv lock`` on the release PR. If these drift apart, the next release PR
would publish a mismatched package, and ``uv sync --locked`` would fail.
"""

import json
import os
import re
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

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


# ── publish.yml ref check ────────────────────────────────────

CHECK_SCRIPT = ROOT / ".github/scripts/check_release_ref.py"
PUBLISH_WORKFLOW = ROOT / ".github/workflows/publish.yml"
TESTPYPI_IF = "github.event_name == 'workflow_dispatch' && github.event.inputs.test_pypi == 'true'"
PYPI_IF = (
    "github.event_name == 'release' || "
    "(github.event_name == 'workflow_dispatch' && github.event.inputs.test_pypi != 'true')"
)
MISMATCH = "does not match package version"
NOT_A_TAG = "Publishing to PyPI needs a run on the"


@pytest.mark.parametrize(
    "event, ref_type, ref_name, test_pypi, version, code, message",
    [
        ("release", "tag", "v1.2.3", None, "1.2.3", 0, None),
        ("release", "tag", "v1.2.4", None, "1.2.3", 1, MISMATCH),
        # release-please dispatches publish.yml on the tag.
        ("workflow_dispatch", "tag", "v1.2.3", "false", "1.2.3", 0, None),
        ("workflow_dispatch", "tag", "v1.2.3", None, "1.2.3", 0, None),
        ("workflow_dispatch", "tag", "v1.2.3", "", "1.2.3", 0, None),
        # A hand run on a branch would publish to PyPI.
        ("workflow_dispatch", "branch", "main", "false", "1.2.3", 1, NOT_A_TAG),
        ("workflow_dispatch", "branch", "main", None, "1.2.3", 1, NOT_A_TAG),
        # A branch named like the tag is still a branch.
        ("workflow_dispatch", "branch", "v1.2.3", "false", "1.2.3", 1, NOT_A_TAG),
        # TestPyPI rehearsal from a branch.
        ("workflow_dispatch", "branch", "feature", "true", "1.2.3", 0, None),
        ("workflow_dispatch", "tag", "v9.9.9", "true", "1.2.3", 1, MISMATCH),
        ("workflow_dispatch", "tag", "v1.2.3", "true", "1.2.3", 0, None),
        # Actions compares case-insensitively but keeps whitespace.
        ("workflow_dispatch", "branch", "main", " true", "1.2.3", 1, NOT_A_TAG),
        ("workflow_dispatch", "branch", "main", "TRUE", "1.2.3", 0, None),
        ("push", "branch", "main", None, "1.2.3", 1, "unexpected event"),
    ],
)
def test_release_ref_check(tmp_path, event, ref_type, ref_name, test_pypi, version, code, message):
    (tmp_path / "pyproject.toml").write_text(f'[project]\nname = "x"\nversion = "{version}"\n')
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("GITHUB_") and key != "TEST_PYPI"
    }
    env.update(GITHUB_EVENT_NAME=event, GITHUB_REF_TYPE=ref_type, GITHUB_REF_NAME=ref_name)
    if test_pypi is not None:
        env["TEST_PYPI"] = test_pypi
    result = subprocess.run(
        [sys.executable, str(CHECK_SCRIPT)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == code, result.stdout + result.stderr
    if message:
        assert "::error::" in result.stdout
        assert message in result.stdout


def _job_block(workflow: str, job: str) -> str:
    match = re.search(rf"^  {re.escape(job)}:\n(.*?)(?=^  \S|\Z)", workflow, re.M | re.S)
    assert match, f"no {job} job"
    return match.group(0)


def test_publish_workflow_runs_the_ref_check():
    workflow = PUBLISH_WORKFLOW.read_text()
    check = _job_block(workflow, "check-ref")
    assert "python .github/scripts/check_release_ref.py" in check
    assert not re.search(r"^\s*(- )?if:", check, re.M)
    assert "TEST_PYPI: ${{ github.event.inputs.test_pypi }}" in check
    assert "github.ref_type == 'tag'" not in workflow
    assert "needs: check-ref" in _job_block(workflow, "test")
    # check_release_ref.py mirrors these conditions; change both together.
    assert f"if: {TESTPYPI_IF}\n" in _job_block(workflow, "publish-to-testpypi")
    assert f"if: {PYPI_IF}\n" in _job_block(workflow, "publish-to-pypi")
    # The ref check relies on release-please dispatching publish.yml on the tag.
    release = (ROOT / ".github/workflows/release-please.yml").read_text()
    assert 'gh workflow run publish.yml --repo "$GITHUB_REPOSITORY" --ref "$TAG"' in release
