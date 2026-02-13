## [0.2.0] - 2026-2-9

### Breaking Changes (v0.1 API still works but is deprecated)
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

## [0.1.3] - 2025-12-31

- Added: CHANGELOG
- Fixed: broken Pypi links

## [0.1.4] - 2026-1-22
- Added: Init adapter for lang graph
- Added: Task support for crew ai adapter
- Added: More unit tests

## [0.1.5] - 2026-2-4
- Added: OpenClaw adapter
- Added: Push notification support
- Added: More unit tests

## [0.1.6] - 2026-2-5
- Added: Add timeout & input mapper for adapters
- Fixed: README fix