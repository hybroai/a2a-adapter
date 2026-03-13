# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [0.2.2] - 2026-03-13

### Added

- GitHub Actions workflow for automated PyPI publishing via trusted publishers
- Multimodality support for N8nAdapter (file and image handling)

### Changed

- N8nAdapter now uses `isinstance` for FilePart check and filters file/image fields from text
- N8nAdapter reuses HTTP client for better performance
- N8nAdapter returns consistent `list[Part]` in multimodal mode
- MIME types are now configurable in N8nAdapter

### Fixed

- Improved Ollama error messages with model name and parsed error detail
- Use consistent `artifact_id` for streaming chunks in executor

### Refactored

- Extracted `_to_parts()` helper in executor for cleaner code

## [0.2.1] - 2026-03-08

### Added

- `OllamaClient` — standalone HTTP client for the Ollama API (`/api/chat`), with streaming support
- `OllamaAdapter` now accepts an `OllamaClient` instance, consistent with how `LangChainAdapter` accepts a runnable and `LangGraphAdapter` accepts a graph
- `OllamaClient` exported from `a2a_adapter` and `a2a_adapter.integrations`
- Convenience constructor preserved: `OllamaAdapter(model="...")` still works for simple cases

### Changed

- `OllamaAdapter.get_metadata()` no longer leaks the model name into AgentCard `name`/`description` defaults

### Refactored

- Separated Ollama HTTP client concerns from the A2A adapter layer for cleaner architecture

## [0.2.0] - 2026-02-09

### Breaking Changes

v0.1 API still works but is deprecated.

- New simplified adapter interface: `BaseA2AAdapter` with single `invoke()` method
- New server functions: `serve_agent(adapter)` and `to_a2a(adapter)` replace manual AgentCard + serve pattern
- New flat imports: `from a2a_adapter import N8nAdapter, serve_agent`

### Added

- `BaseA2AAdapter` — new abstract base with `invoke()` / `stream()` / `cancel()` / `close()`
- `AdapterMetadata` — dataclass for automatic AgentCard generation
- `AdapterAgentExecutor` — bridge layer connecting adapters to A2A SDK
- `server.py` — `to_a2a()`, `serve_agent()`, `build_agent_card()` utilities
- `load_adapter()` — new sync factory function with registry pattern
- `register_adapter()` — decorator for third-party adapter registration
- 6 new v0.2 adapter classes: `N8nAdapter`, `CallableAdapter`, `LangChainAdapter`, `LangGraphAdapter`, `CrewAIAdapter`, `OpenClawAdapter`
- Full streaming support for LangChain and LangGraph adapters
- Lazy imports for all adapter classes (optional deps only loaded on use)
- 117 new unit tests for v0.2 components

### Deprecated

- `BaseAgentAdapter` (v0.1) — use `BaseA2AAdapter`
- `load_a2a_agent()` — use `load_adapter()`
- `build_agent_app()` — use `to_a2a()`
- `client.py` — use `server.py`
- All `*AgentAdapter` classes — use new shorter names (e.g. `N8nAdapter`)

### Design

- SDK-First: delegates task management, SSE streaming, push notifications to A2A SDK
- ~85% code reduction in adapter implementations
- See `DESIGN_V0.2.md` for full architecture rationale

## [0.1.6] - 2026-02-05

### Added

- Timeout & input mapper for adapters

### Fixed

- README fix

## [0.1.5] - 2026-02-04

### Added

- OpenClaw adapter
- Push notification support
- More unit tests

## [0.1.4] - 2026-01-22

### Added

- Init adapter for LangGraph
- Task support for CrewAI adapter
- More unit tests

## [0.1.3] - 2025-12-31

### Added

- CHANGELOG

### Fixed

- Broken PyPI links
