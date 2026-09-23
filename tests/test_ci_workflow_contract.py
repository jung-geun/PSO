"""Static contract for the GitHub Actions workflows.

The CI pipeline stayed red for weeks because of defects that are visible in
the workflow text alone: a setup-uv floating major tag that does not exist,
test jobs that did not install the extras the tests import, a path filter
that skipped ``test/`` changes, and a misspelled publish action.  A distro
Python that ships the stdlib ``test`` package would also have shadowed the
repo's ``test/`` directory.  These checks lock those properties so they cannot
silently regress.

PyYAML is not part of the dev dependency group, so the workflows are parsed as
text.  Every parser below fails closed: if a formatting change makes it find
nothing, the test fails instead of passing vacuously.  The checks cannot tell
whether a pinned tag actually exists upstream (that needs the network); they
only enforce the pin form.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKFLOW_DIR = REPO_ROOT / ".github" / "workflows"
TEST_WORKFLOW = WORKFLOW_DIR / "python-package.yml"
PUBLISH_WORKFLOW = WORKFLOW_DIR / "pypi.yml"

# Every input the test job reads.  tests/ imports modules from test/, this
# contract test reads every workflow file, and the editable build in `uv sync`
# reads README.md and LICENSE (pyproject readme and license-files).
REQUIRED_TEST_PATHS = {
    "pso/**",
    "test/**",
    "tests/**",
    "pyproject.toml",
    "uv.lock",
    ".python-version",
    "README.md",
    "LICENSE",
    ".github/workflows/**",
}

EXACT_TAG = re.compile(r"v\d+\.\d+\.\d+")
FULL_SHA = re.compile(r"[0-9a-f]{40}")


def _workflow_files() -> list[Path]:
    files = sorted([*WORKFLOW_DIR.glob("*.yml"), *WORKFLOW_DIR.glob("*.yaml")])
    assert files, f"no workflow files found under {WORKFLOW_DIR}"
    return files


def _strip_comment(line: str) -> str:
    # Workflow values here never contain '#', so a plain split is enough.
    return line.split("#", 1)[0].rstrip()


def _indent(line: str) -> int:
    return len(line) - len(line.lstrip(" "))


def _top_level_block(text: str, key: str) -> list[str]:
    """Return the lines nested under a top-level ``key:`` mapping."""
    lines = text.splitlines()
    start = None
    for i, raw in enumerate(lines):
        if _strip_comment(raw) == f"{key}:":
            start = i + 1
            break
    assert start is not None, f"top-level '{key}:' not found"
    block = []
    for raw in lines[start:]:
        line = _strip_comment(raw)
        if line and _indent(line) == 0:
            break
        block.append(raw)
    return block


def _children(block: list[str]) -> dict[str, list[str]]:
    """Split a block into its direct child mappings, keyed by name."""
    content = [_strip_comment(raw) for raw in block]
    content = [line for line in content if line.strip()]
    assert content, "empty block"
    child_indent = min(_indent(line) for line in content)
    children: dict[str, list[str]] = {}
    current = None
    for line in content:
        if _indent(line) == child_indent:
            match = re.fullmatch(r"\s*([A-Za-z0-9_-]+):.*", line)
            assert match, f"unexpected line at child level: {line!r}"
            current = match.group(1)
            children[current] = [line]
        else:
            assert current is not None
            children[current].append(line)
    return children


def _trigger_paths(text: str, event: str) -> set[str]:
    triggers = _children(_top_level_block(text, "on"))
    assert event in triggers, f"'{event}' trigger missing"
    paths: set[str] = set()
    in_paths = False
    paths_indent = 0
    for line in triggers[event][1:]:
        if re.fullmatch(r"\s*paths:", line):
            in_paths = True
            paths_indent = _indent(line)
            continue
        if in_paths:
            item = re.fullmatch(r"\s*-\s*[\"']?([^\"']+)[\"']?", line)
            if item and _indent(line) >= paths_indent:
                paths.add(item.group(1).strip())
                continue
            in_paths = False
    assert paths, f"no paths parsed for the '{event}' trigger"
    return paths


def _jobs(text: str) -> dict[str, str]:
    return {
        name: "\n".join(lines)
        for name, lines in _children(_top_level_block(text, "jobs")).items()
    }


def _uses_refs(text: str, action: str) -> list[str]:
    # Tolerates quoted values: `uses: "astral-sh/setup-uv@v9"` must be seen.
    return re.findall(rf"uses:\s*[\"']?{re.escape(action)}@([^\s\"']+)", text)


def _code(text: str) -> str:
    return "\n".join(_strip_comment(raw) for raw in text.splitlines())


def _setup_uv_steps(text: str) -> list[list[str]]:
    """Return the lines of every step that uses astral-sh/setup-uv."""
    lines = [line for line in _code(text).splitlines() if line.strip()]
    steps = []
    for i, line in enumerate(lines):
        if not _uses_refs(line, "astral-sh/setup-uv"):
            continue
        key_indent = line.index("uses:")
        start = i
        while not re.match(rf" {{{key_indent - 2}}}- ", lines[start]):
            start -= 1
            assert start >= 0 and _indent(lines[start]) >= key_indent - 2, (
                f"cannot find the start of the setup-uv step at {line!r}"
            )
        end = i + 1
        while end < len(lines) and _indent(lines[end]) >= key_indent:
            end += 1
        steps.append(lines[start:end])
    return steps


def _input(step: list[str], name: str) -> str | None:
    for line in step:
        match = re.fullmatch(rf"\s*{re.escape(name)}:\s*[\"']?([^\"']*)[\"']?\s*", line)
        if match:
            return match.group(1).strip()
    return None


def test_workflows_use_setup_uv() -> None:
    # Guards the pin check below against passing vacuously; a workflow that
    # does not need uv (labeler, stale bot, ...) is fine on its own.
    refs = [
        ref
        for workflow in _workflow_files()
        for ref in _uses_refs(workflow.read_text(encoding="utf-8"), "astral-sh/setup-uv")
    ]
    assert refs, "no workflow uses astral-sh/setup-uv; update this contract"


@pytest.mark.parametrize("workflow", _workflow_files(), ids=lambda p: p.name)
def test_setup_uv_is_pinned_to_an_exact_release(workflow: Path) -> None:
    code = _code(workflow.read_text(encoding="utf-8"))
    refs = _uses_refs(code, "astral-sh/setup-uv")
    # Fail closed on any reference form the parser does not understand.
    assert len(refs) == code.count("astral-sh/setup-uv"), (
        f"{workflow.name}: could not parse every astral-sh/setup-uv reference"
    )
    for ref in refs:
        assert EXACT_TAG.fullmatch(ref) or FULL_SHA.fullmatch(ref), (
            f"{workflow.name}: astral-sh/setup-uv@{ref} must be an exact vX.Y.Z "
            "tag or a full commit SHA; setup-uv publishes no floating major "
            "tags after v7, so '@v9' fails in 'Set up job'"
        )


@pytest.mark.parametrize("workflow", _workflow_files(), ids=lambda p: p.name)
def test_setup_uv_cache_is_pruned(workflow: Path) -> None:
    # setup-uv caches on GitHub-hosted runners by default ("auto") and, unless
    # prune-cache is set, uploads the whole uv cache: several GB of torch/CUDA
    # wheels per matrix leg against a 10 GB repository cache quota.
    for step in _setup_uv_steps(workflow.read_text(encoding="utf-8")):
        if _input(step, "enable-cache") == "false":
            continue
        assert _input(step, "prune-cache") == "true", (
            f"{workflow.name}: a caching setup-uv step must set 'prune-cache: true'"
        )


def test_test_package_is_a_regular_package() -> None:
    # A namespace package loses to any regular `test` package on sys.path, and
    # Ubuntu's system Python ships one in its stdlib.
    assert (REPO_ROOT / "test" / "__init__.py").is_file(), (
        "test/__init__.py is missing; `from test import ...` in tests/ would "
        "resolve to a distro stdlib `test` package"
    )


def test_test_workflow_token_is_read_only() -> None:
    text = TEST_WORKFLOW.read_text(encoding="utf-8")
    block = [_strip_comment(raw).strip() for raw in _top_level_block(text, "permissions")]
    assert [line for line in block if line] == ["contents: read"], (
        f"{TEST_WORKFLOW.name} must declare top-level 'permissions: contents: read'"
    )


@pytest.mark.parametrize("event", ["push", "pull_request"])
def test_test_workflow_path_filter_covers_every_input(event: str) -> None:
    paths = _trigger_paths(TEST_WORKFLOW.read_text(encoding="utf-8"), event)
    missing = REQUIRED_TEST_PATHS - paths
    assert not missing, f"{TEST_WORKFLOW.name} {event}.paths is missing {sorted(missing)}"


@pytest.mark.parametrize("workflow", [TEST_WORKFLOW, PUBLISH_WORKFLOW], ids=lambda p: p.name)
def test_pytest_jobs_install_the_examples_extra(workflow: Path) -> None:
    jobs = _jobs(workflow.read_text(encoding="utf-8"))
    pytest_jobs = {name: body for name, body in jobs.items() if "pytest" in body}
    assert pytest_jobs, f"{workflow.name} has no job that runs pytest"
    for name, body in pytest_jobs.items():
        syncs = re.findall(r"uv sync\b[^\n]*", body)
        assert syncs, f"{workflow.name}:{name} runs pytest without 'uv sync'"
        for sync in syncs:
            # test/ research modules import matplotlib at module level, and it
            # ships only in the 'examples' extra.
            assert "--extra examples" in sync, (
                f"{workflow.name}:{name}: '{sync}' must include '--extra examples'"
            )


@pytest.mark.parametrize("workflow", [TEST_WORKFLOW, PUBLISH_WORKFLOW], ids=lambda p: p.name)
def test_pytest_jobs_use_uv_managed_python(workflow: Path) -> None:
    jobs = _jobs(workflow.read_text(encoding="utf-8"))
    pytest_jobs = {name: body for name, body in jobs.items() if "pytest" in body}
    assert pytest_jobs, f"{workflow.name} has no job that runs pytest"
    for name, body in pytest_jobs.items():
        # Otherwise uv picks the runner's system Python 3.10, whose stdlib ships
        # a regular `test` package.
        assert re.search(r"^\s*UV_MANAGED_PYTHON:\s*[\"']?1[\"']?$", body, re.MULTILINE), (
            f"{workflow.name}:{name} must set 'UV_MANAGED_PYTHON: \"1\"' in its env"
        )


def test_publish_job_uses_the_real_action_and_waits_for_tests() -> None:
    jobs = _jobs(PUBLISH_WORKFLOW.read_text(encoding="utf-8"))
    assert "publish" in jobs, f"{PUBLISH_WORKFLOW.name} has no 'publish' job"
    publish = jobs["publish"]
    assert re.search(r"^\s*needs:\s*\[?\s*test\s*\]?\s*$", publish, re.MULTILINE), (
        "the publish job must keep 'needs: test' so releases stay gated on tests"
    )
    refs = re.findall(r"uses:\s*[\"']?(pypa/[^@\s\"']+)@([^\s\"']+)", publish)
    assert [action for action, _ in refs] == ["pypa/gh-action-pypi-publish"], (
        f"publish job must use pypa/gh-action-pypi-publish, found {refs}"
    )
    # The job hands a long-lived PyPI token to this action, so the ref must not
    # be a movable branch such as release/v1.
    ref = refs[0][1]
    assert EXACT_TAG.fullmatch(ref) or FULL_SHA.fullmatch(ref), (
        f"pypa/gh-action-pypi-publish@{ref} must be an exact vX.Y.Z tag or a "
        "full commit SHA"
    )
