"""Check that a publish.yml run is on the release tag before anything is built.

publish.yml publishes to PyPI on a ``release`` event, or on a hand run
(``workflow_dispatch``) unless ``test_pypi`` is true. Such a run must be on the
``v<version>`` tag for the version in pyproject.toml. A TestPyPI rehearsal may
run from a branch; on a tag, the tag must still match the version.

The publish job conditions in publish.yml decide what a run publishes, and this
script mirrors them. tests/unit/test_release_config.py pins both; change them
together.
"""

import os
import sys
import tomllib


def _error(message: str) -> int:
    print(f"::error::{message}")
    return 1


def main() -> int:
    event = os.environ.get("GITHUB_EVENT_NAME", "")
    ref_type = os.environ.get("GITHUB_REF_TYPE", "")
    ref_name = os.environ.get("GITHUB_REF_NAME", "")
    # Actions compares strings in ``if:`` ignoring case but not whitespace, so
    # " true" still publishes to PyPI. No strip(): fail closed.
    test_pypi = os.environ.get("TEST_PYPI", "").lower() == "true"

    with open("pyproject.toml", "rb") as f:
        version = tomllib.load(f)["project"]["version"]
    tag = f"v{version}"

    if event == "release" or (event == "workflow_dispatch" and not test_pypi):
        if ref_type != "tag":
            return _error(
                f"Publishing to PyPI needs a run on the {tag} tag, not {ref_type} "
                f"{ref_name}; run from a branch only with test_pypi=true."
            )
    elif event == "workflow_dispatch":
        if ref_type != "tag":
            print(f"TestPyPI rehearsal from {ref_type} {ref_name}.")
            return 0
    else:
        return _error(f"publish.yml ran on an unexpected event: {event or '(none)'}.")

    if ref_name != tag:
        return _error(
            f"Release tag {ref_name} does not match package version {tag}. Release "
            "through the release-please PR so the version is bumped first."
        )
    print(f"Ref {ref_name} matches package version {tag}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
