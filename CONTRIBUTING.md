# Contributing to A2A Adapters

Thank you for your interest in contributing to the A2A Adapters project! This document provides guidelines and instructions for contributing.

## Code of Conduct

This project adheres to a code of conduct. By participating, you are expected to uphold this code. Please be respectful and considerate in your interactions.

## How to Contribute

### Reporting Bugs

1. Check if the bug has already been reported in [Issues](https://github.com/hybro-ai/a2a-adapter/issues)
2. If not, create a new issue with:
   - Clear title and description
   - Steps to reproduce
   - Expected vs actual behavior
   - Environment details (Python version, OS, etc.)
   - Code samples if applicable

### Suggesting Features

1. Check [Issues](https://github.com/hybro-ai/a2a-adapter/issues) for existing feature requests
2. Create a new issue with:
   - Clear use case description
   - Proposed API or implementation approach
   - Potential impact on existing code

### Adding a New Framework Adapter

We welcome adapters for new agent frameworks! Here's how to add one:

#### 1. Create the Adapter File

Create `a2a_adapter/integrations/{framework}.py`:

```python
"""
{Framework} adapter for A2A Protocol.
"""

from typing import Any, Dict
from a2a.types import Message, MessageSendParams, TextPart
from ..adapter import BaseAgentAdapter


class {Framework}AgentAdapter(BaseAgentAdapter):
    """
    Adapter for integrating {Framework} with A2A Protocol.
    """

    def __init__(self, ...):
        # Initialize with framework-specific config
        pass

    async def to_framework(self, params: MessageSendParams) -> Any:
        # Convert A2A params to framework input
        pass

    async def call_framework(
        self, framework_input: Any, params: MessageSendParams
    ) -> Any:
        # Call the framework
        pass

    async def from_framework(
        self, framework_output: Any, params: MessageSendParams
    ) -> Message:
        # Convert framework output to A2A Message
        pass
    
    # Optional: Add handle_stream() if framework supports streaming
```

#### 2. Update the Loader

Add your adapter to `a2a_adapter/loader.py`:

```python
elif adapter_type == "{framework}":
    from .integrations.{framework} import {Framework}AgentAdapter
    
    # Validate required config
    required_param = config.get("required_param")
    if not required_param:
        raise ValueError("{framework} adapter requires 'required_param' in config")
    
    return {Framework}AgentAdapter(
        required_param=required_param,
        optional_param=config.get("optional_param", default_value),
    )
```

#### 3. Update Integrations __init__

Add to `a2a_adapter/integrations/__init__.py`:

```python
__all__ = [
    ...,
    "{Framework}AgentAdapter",
]

def __getattr__(name: str):
    ...
    elif name == "{Framework}AgentAdapter":
        from .{framework} import {Framework}AgentAdapter
        return {Framework}AgentAdapter
    ...
```

#### 4. Update pyproject.toml

Add optional dependency:

```toml
[project.optional-dependencies]
{framework} = ["{framework}>=X.Y.Z"]
```

#### 5. Create an Example

Create `examples/0X_{framework}_agent.py`:

```python
"""
Example: Single {Framework} Agent Server
"""

import asyncio
from a2a_adapter import load_a2a_agent, serve_agent
from a2a.types import AgentCard

async def main():
    adapter = await load_a2a_agent({
        "adapter": "{framework}",
        # ... config ...
    })
    
    card = AgentCard(
        name="{Framework} Agent",
        description="...",
    )
    
    serve_agent(agent_card=card, adapter=adapter, port=800X)

if __name__ == "__main__":
    asyncio.run(main())
```

#### 6. Add Tests

Create `tests/unit/test_{framework}_adapter.py`:

```python
"""
Unit tests for {Framework}AgentAdapter.
"""

import pytest
from a2a_adapter.integrations.{framework} import {Framework}AgentAdapter
from a2a.types import Message, MessageSendParams, TextPart


@pytest.mark.asyncio
async def test_{framework}_adapter_basic():
    # Test basic functionality
    pass
```

#### 7. Update Documentation

- Add row to framework support table in README.md
- Document configuration options
- Add to loader documentation

### Submitting Pull Requests

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests (`pytest`)
5. Run linters (`black .`, `ruff check .`)
6. Commit with clear messages (`git commit -m 'Add amazing feature'`)
7. Push to your fork (`git push origin feature/amazing-feature`)
8. Open a Pull Request

#### PR Guidelines

- Keep changes focused and atomic
- Include tests for new functionality
- Update documentation as needed
- Ensure all tests pass
- Follow existing code style
- Write clear commit messages

## Development Setup

### 1. Clone the Repository

```bash
git clone https://github.com/hybro-ai/a2a-adapter.git
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
│   ├── __init__.py         # Package exports
│   ├── adapter.py          # BaseAgentAdapter
│   ├── loader.py           # Adapter factory
│   ├── client.py           # Server helpers
│   └── integrations/       # Framework adapters
│       ├── n8n.py
│       ├── crewai.py
│       ├── langchain.py
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

Follow conventional commits format:

```
type(scope): brief description

Longer explanation if needed

Fixes #123
```

Types: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`

Examples:
- `feat(langchain): add streaming support`
- `fix(n8n): handle timeout errors properly`
- `docs(readme): update installation instructions`

## Release Process

(For maintainers)

1. Update version in `pyproject.toml` and `a2a_adapter/__init__.py`
2. Update CHANGELOG.md
3. Create git tag: `git tag v0.1.0`
4. Push tag: `git push origin v0.1.0`
5. Build package: `python -m build`
6. Upload to PyPI: `twine upload dist/*`

## Getting Help

- 📚 Read the [README](README.md) and [ARCHITECTURE](ARCHITECTURE.md)
- 💬 Join [Discussions](https://github.com/hybro-ai/a2a-adapter/discussions)
- 🐛 Check [Issues](https://github.com/hybro-ai/a2a-adapter/issues)
- 📧 Contact maintainers

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing to A2A Adapters! 🎉

