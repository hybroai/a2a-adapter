# Contributing to A2A Adapters

Thank you for your interest in contributing to the A2A Adapters project! 🎉

This document provides guidelines and instructions for contributing. Whether you're fixing bugs, adding features, improving documentation, or creating new adapters, your contributions are welcome and appreciated!

## 📋 Table of Contents

- [Code of Conduct](#code-of-conduct)
- [How to Contribute](#how-to-contribute)
- [Development Setup](#development-setup)
- [Coding Standards](#coding-standards)
- [Pull Request Process](#pull-request-process)
- [Adding New Adapters](#adding-a-new-framework-adapter)
- [Release Process](#release-process)

## Code of Conduct

This project adheres to a code of conduct. By participating, you are expected to:

- Be respectful and considerate in your interactions
- Welcome newcomers and help them learn
- Focus on what is best for the community
- Show empathy towards other community members

We are committed to providing a welcoming and inspiring community for all.

## How to Contribute

### Reporting Bugs

**Before reporting a bug:**

1. ✅ Check if the bug has already been reported in [Issues](https://github.com/hybroai/a2a-adapter/issues)
2. ✅ Search closed issues - it might have been fixed already
3. ✅ Try to reproduce the bug with the latest version

**When creating a bug report, include:**

- **Clear title** - Summarize the issue in one line
- **Description** - What happened vs what you expected
- **Steps to reproduce** - Minimal code example that demonstrates the bug
- **Environment details**:
  - Python version (`python --version`)
  - OS and version
  - Package version (`pip show a2a-adapter`)
  - Framework versions (if applicable)
- **Error messages** - Full traceback if available
- **Screenshots** - If applicable

**Example bug report:**

````markdown
**Describe the bug**
The n8n adapter times out after 10 seconds even when timeout is set to 60.

**To Reproduce**

```python
adapter = await load_a2a_agent({
    "adapter": "n8n",
    "webhook_url": "...",
    "timeout": 60
})
```
````

**Expected behavior**
Should wait up to 60 seconds before timing out.

**Environment**

- Python 3.11.5
- macOS 14.0
- a2a-adapter 0.1.0

````

### Suggesting Features

**Before suggesting a feature:**

1. ✅ Check [Issues](https://github.com/hybroai/a2a-adapter/issues) for existing feature requests
2. ✅ Consider if it fits the project's scope (A2A protocol adapter SDK)
3. ✅ Think about the API design and backward compatibility

**When suggesting a feature, include:**

- **Use case** - Why is this feature needed?
- **Proposed API** - How would users interact with it?
- **Implementation approach** - High-level design (optional)
- **Alternatives considered** - Other ways to solve the problem
- **Impact** - Breaking changes? New dependencies?

**Example feature request:**

```markdown
**Feature: AutoGen Adapter**

**Use case**
Enable AutoGen multi-agent systems to communicate via A2A protocol.

**Proposed API**
```python
adapter = await load_a2a_agent({
    "adapter": "autogen",
    "group_chat": autogen_group_chat_instance
})
````

**Benefits**

- Enables AutoGen agents in A2A ecosystems
- Follows existing adapter pattern

````

### Adding a New Framework Adapter

We welcome adapters for new agent frameworks! Here's how to add one:

#### 1. Create the Adapter File

Create `a2a_adapter/integrations/{framework}.py`:

```python
"""
{Framework} adapter for A2A Protocol.
"""

from ..base_adapter import AdapterMetadata, BaseA2AAdapter


class {Framework}Adapter(BaseA2AAdapter):
    """
    Adapter for integrating {Framework} with A2A Protocol.
    """

    def __init__(self, ..., name="", description=""):
        # Initialize with framework-specific config
        self._name = name
        self._description = description

    async def invoke(self, user_input: str, context_id: str | None = None, **kwargs) -> str:
        # Call the framework and return a text response
        result = await call_my_framework(user_input)
        return str(result)

    # Optional: streaming support
    async def stream(self, user_input: str, context_id: str | None = None, **kwargs):
        async for chunk in my_framework_stream(user_input):
            yield str(chunk)

    # Optional: metadata for AgentCard
    def get_metadata(self) -> AdapterMetadata:
        return AdapterMetadata(
            name=self._name or "{Framework}Adapter",
            description=self._description,
            streaming=self.supports_streaming(),
        )

    # Optional: resource cleanup
    async def close(self) -> None:
        await self._client.aclose()
````

#### 2. Update the Loader

Add your adapter to the `_BUILTIN_MAP` in `a2a_adapter/loader.py`:

```python
_BUILTIN_MAP = {
    ...
    "{framework}": ("a2a_adapter.integrations.{framework}", "{Framework}Adapter"),
}
```

#### 3. Update Integrations __init__

Add to `a2a_adapter/integrations/__init__.py`:

```python
__all__ = [
    ...,
    "{Framework}Adapter",
]

def __getattr__(name: str):
    ...
    elif name == "{Framework}Adapter":
        from .{framework} import {Framework}Adapter
        return {Framework}Adapter
    ...
```

#### 4. Update pyproject.toml

Add optional dependency:

```toml
[project.optional-dependencies]
{framework} = ["{framework}>=X.Y.Z"]
```

#### 5. Create an Example

Create `examples/{framework}_agent.py`:

```python
"""
Example: {Framework} Agent Server
"""

from a2a_adapter import {Framework}Adapter, serve_agent

adapter = {Framework}Adapter(
    # ... config ...
    name="{Framework} Agent",
    description="...",
)
serve_agent(adapter, port=800X)
```

#### 6. Add Tests

Create `tests/unit/test_{framework}_adapter.py`:

```python
"""
Unit tests for {Framework}Adapter.
"""

import pytest
from a2a_adapter.integrations.{framework} import {Framework}Adapter


class TestInvoke:
    @pytest.mark.asyncio
    async def test_basic(self):
        adapter = {Framework}Adapter(...)
        result = await adapter.invoke("Hello")
        assert isinstance(result, str)


class TestMetadata:
    def test_default_metadata(self):
        adapter = {Framework}Adapter(...)
        meta = adapter.get_metadata()
        assert meta.name
```

#### 7. Update Documentation

- Add row to framework support table in README.md
- Document configuration options
- Add to loader documentation

### Submitting Pull Requests

**Before submitting:**

1. ✅ Fork the repository
2. ✅ Create a feature branch (`git checkout -b feature/amazing-feature`)
3. ✅ Make your changes
4. ✅ Run tests (`pytest`)
5. ✅ Run linters (`black .`, `ruff check .`)
6. ✅ Update documentation if needed
7. ✅ Ensure all tests pass

**PR Process:**

1. **Push to your fork** (`git push origin feature/amazing-feature`)
2. **Open a Pull Request** on GitHub
3. **Fill out the PR template** (see below)
4. **Wait for review** - Maintainers will review your PR
5. **Address feedback** - Make requested changes
6. **Get approval** - Once approved, your PR will be merged!

#### PR Template

When opening a PR, use this template:

```markdown
## Description

Brief description of what this PR does.

## Type of Change

- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Related Issues

Fixes #123
Related to #456

## Testing

- [ ] Added tests for new functionality
- [ ] All existing tests pass
- [ ] Tested manually with examples

## Checklist

- [ ] Code follows project style guidelines
- [ ] Self-review completed
- [ ] Comments added for complex code
- [ ] Documentation updated
- [ ] No new warnings generated
- [ ] Tests added/updated
```

#### PR Guidelines

- ✅ **Keep changes focused** - One feature/fix per PR
- ✅ **Include tests** - New functionality must have tests
- ✅ **Update docs** - README, docstrings, or examples
- ✅ **Ensure tests pass** - All CI checks must pass
- ✅ **Follow code style** - Use Black, Ruff, and type hints
- ✅ **Write clear commits** - Use conventional commit format
- ✅ **Keep PRs small** - Easier to review and merge

## Development Setup

### 1. Clone the Repository

```bash
git clone git@github.com:hybroai/a2a-adapter.git
cd a2a-adapter
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install in Development Mode

```bash
# Install package in editable mode with all dependencies
pip install -e ".[all,dev]"
```

### 4. Run Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=a2a_adapter --cov-report=html

# Run specific test file
pytest tests/unit/test_adapter.py

# Run with verbose output
pytest -v
```

### 5. Code Formatting

```bash
# Format code with Black
black a2a_adapter/ examples/ tests/

# Check with Ruff
ruff check a2a_adapter/ examples/ tests/

# Type checking with mypy
mypy a2a_adapter/
```

## Project Structure

```
a2a-adapter/
├── a2a_adapter/           # Main package
│   ├── __init__.py         # Package exports + lazy imports
│   ├── base_adapter.py     # BaseA2AAdapter + AdapterMetadata
│   ├── executor.py         # AdapterAgentExecutor (bridge)
│   ├── server.py           # to_a2a() / serve_agent() / build_agent_card()
│   ├── loader.py           # load_adapter() / register_adapter()
│   ├── adapter.py          # [deprecated] v0.1 BaseAgentAdapter
│   ├── client.py           # [deprecated] v0.1 server helpers
│   └── integrations/       # Framework adapters
│       ├── n8n.py
│       ├── crewai.py
│       ├── langchain.py
│       ├── langgraph.py
│       ├── ollama.py        # OllamaClient + OllamaAdapter
│       ├── openclaw.py
│       └── callable.py
├── examples/               # Usage examples
├── tests/                  # Test suite
│   ├── unit/              # Unit tests
│   └── integration/       # Integration tests
├── pyproject.toml         # Package configuration
├── README.md              # User documentation
├── ARCHITECTURE.md        # Technical documentation
└── CONTRIBUTING.md        # This file
```

## Coding Standards

### Python Style

- Follow PEP 8
- Use Black for formatting (line length: 100)
- Use type hints where possible
- Write docstrings for public APIs (Google style)

### Documentation Style

```python
def function_name(param1: str, param2: int) -> bool:
    """
    Short description of function.

    Longer description if needed, explaining the purpose,
    behavior, and any important details.

    Args:
        param1: Description of param1
        param2: Description of param2

    Returns:
        Description of return value

    Raises:
        ValueError: When something goes wrong

    Example:
        >>> result = function_name("test", 42)
        >>> print(result)
        True
    """
    pass
```

### Testing Guidelines

- Write tests for all new functionality
- Aim for >80% code coverage
- Use pytest fixtures for common setup
- Mock external dependencies (HTTP calls, framework APIs)
- Test both success and error cases

### Commit Messages

**Follow [Conventional Commits](https://www.conventionalcommits.org/) format:**

```
type(scope): brief description

Longer explanation if needed

Fixes #123
```

**Commit Types:**

- `feat` - New feature
- `fix` - Bug fix
- `docs` - Documentation changes
- `style` - Code style changes (formatting, etc.)
- `refactor` - Code refactoring
- `test` - Adding or updating tests
- `chore` - Maintenance tasks
- `perf` - Performance improvements
- `ci` - CI/CD changes

**Scopes:** `n8n`, `crewai`, `langchain`, `callable`, `loader`, `client`, `docs`, `tests`, etc.

**Good Examples:**

```bash
feat(langchain): add streaming support
fix(n8n): handle timeout errors properly
docs(readme): update installation instructions
test(adapter): add tests for error handling
refactor(loader): simplify adapter loading logic
```

**Bad Examples:**

```bash
# Too vague
fix: bug fix
update: changes
# Missing scope
feat: add new feature
# Not following format
Fixed the bug in n8n adapter
```

## Release Process

**(For maintainers only)**

### Pre-Release Checklist

- [ ] All tests passing
- [ ] Documentation updated
- [ ] CHANGELOG.md updated
- [ ] Version bumped in `pyproject.toml` and `a2a_adapter/__init__.py`
- [ ] Release notes prepared

### Release Steps

1. **Update version**

   ```bash
   # Update pyproject.toml
   version = "0.1.1"

   # Update a2a_adapter/__init__.py
   __version__ = "0.1.1"
   ```

2. **Update CHANGELOG.md**

   ```markdown
   ## [0.1.1] - 2024-01-15

   - Added: New feature X
   - Fixed: Bug Y
   ```

3. **Create release commit**

   ```bash
   git add pyproject.toml a2a_adapter/__init__.py CHANGELOG.md
   git commit -m "chore: release v0.1.1"
   ```

4. **Create and push tag**

   ```bash
   git tag v0.1.1
   git push origin main --tags
   ```

5. **Build and publish to PyPI**

   ```bash
   python -m build
   twine upload dist/*
   ```

6. **Create GitHub Release**
   - Go to GitHub Releases
   - Create new release from tag
   - Copy CHANGELOG entry
   - Publish release

### Versioning

We follow [Semantic Versioning](https://semver.org/):

- **MAJOR** - Breaking changes
- **MINOR** - New features (backward compatible)
- **PATCH** - Bug fixes (backward compatible)

## 🆘 Getting Help

**Need help? We're here for you!**

- 📚 **Documentation** - Read [README.md](README.md) and [ARCHITECTURE.md](ARCHITECTURE.md)
- 🐛 **Issues** - Check [existing issues](https://github.com/hybroai/a2a-adapter/issues) or create a new one
- 💬 **Discussions** - Ask questions in [GitHub Discussions](https://github.com/hybroai/a2a-adapter/discussions)
- 📧 **Contact** - Reach out to maintainers via GitHub

## 📝 Issue and PR Templates

We recommend creating GitHub issue and PR templates for better organization:

### Issue Template (`.github/ISSUE_TEMPLATE/bug_report.md`)

```markdown
**Describe the bug**
A clear description of what the bug is.

**To Reproduce**
Steps to reproduce the behavior.

**Expected behavior**
What you expected to happen.

**Environment**

- Python version:
- OS:
- Package version:
- Framework versions:

**Additional context**
Any other relevant information.
```

### Feature Request Template (`.github/ISSUE_TEMPLATE/feature_request.md`)

```markdown
**Is your feature request related to a problem?**
A clear description of the problem.

**Describe the solution you'd like**
A clear description of what you want to happen.

**Describe alternatives you've considered**
Other solutions or features you've considered.

**Additional context**
Any other relevant information.
```

## 📄 License

By contributing, you agree that your contributions will be licensed under the Apache License 2.0.

## 🙏 Recognition

Contributors will be:

- Listed in the README (if desired)
- Credited in release notes
- Appreciated by the community! 🎉

---

**Thank you for contributing to A2A Adapters!** 🎉

Your contributions make this project better for everyone. We appreciate your time and effort!
