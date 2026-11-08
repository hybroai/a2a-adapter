"""Tests for the v0.2 CrewAIAdapter.

Covers: invoke (async + sync fallback), input building, output extraction,
        metadata, and integration with to_a2a / build_agent_card / load_adapter.
"""

import asyncio
import json
import logging
import threading
from types import SimpleNamespace

import pytest

from a2a_adapter.integrations.crewai import CrewAIAdapter
from a2a_adapter.server import build_agent_card, to_a2a
from a2a_adapter.loader import load_adapter


# ──── Helpers: Mock Crews ────


class MockCrewOutput:
    """Simulates a CrewAI CrewOutput object with .raw attribute."""

    def __init__(self, raw):
        self.raw = raw


class MockCrewOutputResult:
    """Simulates a CrewOutput with .result (older API)."""

    def __init__(self, result):
        self.result = result


class MockCrew:
    """Mock CrewAI Crew with kickoff_async."""

    def __init__(self, response="crew result"):
        self._response = response
        self.last_inputs = None

    async def kickoff_async(self, inputs=None):
        self.last_inputs = inputs
        return MockCrewOutput(self._response)


class MockCrewSyncOnly:
    """Mock crew without kickoff_async (older CrewAI version)."""

    def __init__(self, response="sync result"):
        self._response = response
        self.last_inputs = None

    def kickoff(self, inputs=None):
        self.last_inputs = inputs
        return MockCrewOutput(self._response)


class MockCrewReturnsDict:
    """Mock crew that returns a dict."""

    async def kickoff_async(self, inputs=None):
        return {"output": "from dict", "extra": "ignored"}


class MockCrewReturnsResult:
    """Mock crew that returns CrewOutput with .result."""

    async def kickoff_async(self, inputs=None):
        return MockCrewOutputResult("from result attr")


# ──── Test: Invoke ────


@pytest.mark.asyncio
async def test_invoke_basic():
    adapter = CrewAIAdapter(crew=MockCrew("hello world"))
    result = await adapter.invoke("test input")
    assert result == "hello world"


@pytest.mark.asyncio
async def test_invoke_passes_inputs():
    crew = MockCrew()
    adapter = CrewAIAdapter(crew=crew)
    await adapter.invoke("run this")
    assert crew.last_inputs == {"inputs": "run this"}


@pytest.mark.asyncio
async def test_invoke_sync_fallback():
    """Falls back to sync kickoff() when kickoff_async is missing."""
    crew = MockCrewSyncOnly("sync response")
    adapter = CrewAIAdapter(crew=crew)
    result = await adapter.invoke("test")
    assert result == "sync response"
    assert crew.last_inputs == {"inputs": "test"}


@pytest.mark.asyncio
async def test_invoke_dict_output():
    adapter = CrewAIAdapter(crew=MockCrewReturnsDict())
    result = await adapter.invoke("test")
    assert result == "from dict"


@pytest.mark.asyncio
async def test_invoke_result_attr():
    adapter = CrewAIAdapter(crew=MockCrewReturnsResult())
    result = await adapter.invoke("test")
    assert result == "from result attr"


# ──── Test: Input Building ────


@pytest.mark.asyncio
async def test_input_key_custom():
    crew = MockCrew()
    adapter = CrewAIAdapter(crew=crew, inputs_key="topic")
    await adapter.invoke("machine learning")
    assert crew.last_inputs == {"topic": "machine learning"}


@pytest.mark.asyncio
async def test_input_json_parsing():
    crew = MockCrew()
    adapter = CrewAIAdapter(crew=crew, parse_json_input=True)
    await adapter.invoke('{"customer_domain": "example.com", "topic": "AI"}')
    assert crew.last_inputs == {"customer_domain": "example.com", "topic": "AI"}


@pytest.mark.asyncio
async def test_input_json_disabled():
    crew = MockCrew()
    adapter = CrewAIAdapter(crew=crew, parse_json_input=False)
    json_str = '{"key": "value"}'
    await adapter.invoke(json_str)
    assert crew.last_inputs == {"inputs": json_str}


@pytest.mark.asyncio
async def test_input_mapper():
    crew = MockCrew()

    def mapper(raw_input, context_id):
        return {"customer_domain": raw_input, "ctx": context_id or "none"}

    adapter = CrewAIAdapter(crew=crew, input_mapper=mapper)
    await adapter.invoke("example.com", context_id="ctx-1")
    assert crew.last_inputs == {"customer_domain": "example.com", "ctx": "ctx-1"}


@pytest.mark.asyncio
async def test_input_mapper_error_fails_the_task():
    """A bug in input_mapper must surface, not silently fall back."""
    crew = MockCrew()

    def bad_mapper(raw_input, context_id):
        raise ValueError("broken")

    adapter = CrewAIAdapter(crew=crew, input_mapper=bad_mapper)
    with pytest.raises(RuntimeError, match="input_mapper failed: ValueError: broken") as exc:
        await adapter.invoke("hello")
    assert isinstance(exc.value.__cause__, ValueError)
    assert crew.last_inputs is None


@pytest.mark.asyncio
async def test_input_mapper_must_return_a_dict():
    crew = MockCrew()
    adapter = CrewAIAdapter(crew=crew, input_mapper=lambda raw, ctx: ["not", "a", "dict"])
    with pytest.raises(TypeError, match="input_mapper must return a dict, got list"):
        await adapter.invoke("hello")


@pytest.mark.asyncio
async def test_default_inputs_merged():
    crew = MockCrew()
    adapter = CrewAIAdapter(
        crew=crew,
        default_inputs={"language": "en"},
    )
    await adapter.invoke("test")
    assert crew.last_inputs == {"language": "en", "inputs": "test"}


@pytest.mark.asyncio
async def test_default_inputs_overridden_by_json():
    crew = MockCrew()
    adapter = CrewAIAdapter(
        crew=crew,
        default_inputs={"language": "en"},
    )
    await adapter.invoke('{"language": "zh", "topic": "AI"}')
    assert crew.last_inputs == {"language": "zh", "topic": "AI"}


# ──── Test: Streaming NOT supported ────


def test_no_streaming():
    adapter = CrewAIAdapter(crew=MockCrew())
    assert adapter.supports_streaming() is False


# ──── Test: Metadata ────


def test_metadata_defaults():
    adapter = CrewAIAdapter(crew=MockCrew())
    meta = adapter.get_metadata()
    assert meta.name == "CrewAIAdapter"
    assert meta.streaming is False


def test_metadata_custom():
    adapter = CrewAIAdapter(
        crew=MockCrew(),
        name="Research Crew",
        description="Researches topics using multiple agents",
    )
    meta = adapter.get_metadata()
    assert meta.name == "Research Crew"
    assert meta.description == "Researches topics using multiple agents"


# ──── Test: Output Edge Cases ────


@pytest.mark.asyncio
async def test_output_string():
    class StringCrew:
        async def kickoff_async(self, inputs=None):
            return "plain string"

    adapter = CrewAIAdapter(crew=StringCrew())
    result = await adapter.invoke("test")
    assert result == "plain string"


@pytest.mark.asyncio
async def test_output_dict_fallback_json():
    class WeirdDictCrew:
        async def kickoff_async(self, inputs=None):
            return {"custom_key": "custom_value"}

    adapter = CrewAIAdapter(crew=WeirdDictCrew())
    result = await adapter.invoke("test")
    assert json.loads(result) == {"custom_key": "custom_value"}


@pytest.mark.asyncio
async def test_output_dict_common_keys():
    """Test extraction via common output keys."""
    for key in ("output", "result", "response", "answer", "text"):

        class DictCrew:
            _key = key

            async def kickoff_async(self, inputs=None):
                return {self._key: f"from_{self._key}"}

        adapter = CrewAIAdapter(crew=DictCrew())
        result = await adapter.invoke("test")
        assert result == f"from_{key}"


# ──── Test: Integration with server layer ────


def test_build_agent_card():
    adapter = CrewAIAdapter(
        crew=MockCrew(),
        name="Test Crew",
        description="A test crew",
    )
    card = build_agent_card(adapter, url="http://localhost:9004")
    assert card.name == "Test Crew"
    assert card.description == "A test crew"
    assert card.capabilities.streaming is False
    assert card.supported_interfaces[0].url == "http://localhost:9004"


def test_to_a2a_builds_app():
    adapter = CrewAIAdapter(crew=MockCrew())
    app = to_a2a(adapter)
    assert app is not None


def test_load_adapter_crewai():
    adapter = load_adapter(
        {
            "adapter": "crewai",
            "crew": MockCrew(),
            "inputs_key": "topic",
            "timeout": 600,
        }
    )
    assert isinstance(adapter, CrewAIAdapter)
    assert adapter.inputs_key == "topic"
    assert adapter.timeout == 600


# ──── Test: Flat import ────


def test_flat_import():
    from a2a_adapter import CrewAIAdapter as CA

    assert CA is CrewAIAdapter


# ──── Test: kickoff selection and timeout ────


class MockCrewAttributeErrorInside:
    """kickoff_async exists but the crew's own code raises AttributeError."""

    def __init__(self):
        self.kickoff_calls = 0

    async def kickoff_async(self, inputs=None):
        raise AttributeError("'NoneType' object has no attribute 'run'")

    def kickoff(self, inputs=None):
        self.kickoff_calls += 1
        return MockCrewOutput("ran twice")


@pytest.mark.asyncio
async def test_attribute_error_inside_crew_does_not_rerun():
    crew = MockCrewAttributeErrorInside()
    adapter = CrewAIAdapter(crew=crew)
    with pytest.raises(AttributeError, match="no attribute 'run'"):
        await adapter.invoke("test")
    assert crew.kickoff_calls == 0


@pytest.mark.asyncio
async def test_sync_fallback_runs_off_the_event_loop():
    main_thread = threading.get_ident()
    seen = {}

    class Crew:
        def kickoff(self, inputs=None):
            seen["thread"] = threading.get_ident()
            return MockCrewOutput("ok")

    assert await CrewAIAdapter(crew=Crew()).invoke("x") == "ok"
    assert seen["thread"] != main_thread


@pytest.mark.asyncio
async def test_invoke_timeout_has_clear_message():
    class SlowCrew:
        async def kickoff_async(self, inputs=None):
            await asyncio.sleep(1)

    adapter = CrewAIAdapter(crew=SlowCrew(), timeout=0.05)
    with pytest.raises(TimeoutError, match="CrewAI crew timed out after 0.05 seconds"):
        await adapter.invoke("x")


@pytest.mark.asyncio
async def test_output_dict_with_non_json_values():
    from datetime import datetime

    when = datetime(2026, 1, 2, 3, 4, 5)

    class Crew:
        async def kickoff_async(self, inputs=None):
            return {"when": when}

    result = await CrewAIAdapter(crew=Crew()).invoke("test")
    assert json.loads(result) == {"when": str(when)}


# ──── Test: requests are isolated from each other ────

_CONCURRENT_REQUESTS = 4  # low enough for the default executor on small CI machines


class TemplatedCrew:
    """Mutates itself per kickoff, like a real Crew, then waits for every request."""

    def __init__(self, barrier, copies=None):
        self.barrier = barrier
        self.copies = [] if copies is None else copies
        self.rendered = None
        self.kickoffs = 0

    def copy(self):
        clone = type(self)(self.barrier, self.copies)
        self.copies.append(clone)
        return clone

    async def kickoff_async(self, inputs=None):
        self.kickoffs += 1
        self.rendered = f"answer for {inputs['inputs']}"
        await self.barrier.wait()
        return self.rendered


class TemplatedSyncCrew:
    """The same, for crews with only a blocking ``kickoff``."""

    def __init__(self, barrier, copies=None):
        self.barrier = barrier
        self.copies = [] if copies is None else copies
        self.rendered = None
        self.kickoffs = 0

    def copy(self):
        clone = type(self)(self.barrier, self.copies)
        self.copies.append(clone)
        return clone

    def kickoff(self, inputs=None):
        self.kickoffs += 1
        self.rendered = f"answer for {inputs['inputs']}"
        self.barrier.wait(timeout=5)
        return self.rendered


class InFlightCrew:
    """No ``copy``; records how many kickoffs overlap (never waits on a barrier)."""

    def __init__(self):
        self.in_flight = 0
        self.max_in_flight = 0
        self.kickoffs = 0

    async def kickoff_async(self, inputs=None):
        self.kickoffs += 1
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        for _ in range(10):
            await asyncio.sleep(0)
        self.in_flight -= 1
        return inputs["inputs"]


class CountingCopyCrew(InFlightCrew):
    """Has ``copy``, which the adapter must not use (or must stop using)."""

    def __init__(self, **attrs):
        super().__init__()
        self.copy_calls = 0
        for name, value in attrs.items():
            setattr(self, name, value)

    def copy(self):
        self.copy_calls += 1
        return type(self)()


async def _serve_concurrently(adapter, count=_CONCURRENT_REQUESTS):
    prompts = [f"request-{i}" for i in range(count)]
    calls = [adapter.invoke(prompt) for prompt in prompts]
    return prompts, await asyncio.wait_for(asyncio.gather(*calls), 10)


@pytest.mark.asyncio
async def test_concurrent_requests_each_get_their_own_input():
    crew = TemplatedCrew(asyncio.Barrier(_CONCURRENT_REQUESTS))
    prompts, results = await _serve_concurrently(CrewAIAdapter(crew=crew))
    assert results == [f"answer for {prompt}" for prompt in prompts]


@pytest.mark.asyncio
async def test_concurrent_requests_sync_only_crew():
    crew = TemplatedSyncCrew(threading.Barrier(_CONCURRENT_REQUESTS))
    prompts, results = await _serve_concurrently(CrewAIAdapter(crew=crew))
    assert results == [f"answer for {prompt}" for prompt in prompts]


@pytest.mark.asyncio
async def test_each_request_copies_and_leaves_the_original_untouched():
    crew = TemplatedCrew(asyncio.Barrier(_CONCURRENT_REQUESTS))
    await _serve_concurrently(CrewAIAdapter(crew=crew))
    assert len(crew.copies) == _CONCURRENT_REQUESTS
    assert all(clone.kickoffs == 1 for clone in crew.copies)
    assert crew.kickoffs == 0
    assert crew.rendered is None


@pytest.mark.asyncio
async def test_crew_without_copy_is_serialised_and_warns_once(caplog):
    crew = InFlightCrew()
    with caplog.at_level(logging.WARNING, logger="a2a_adapter.integrations.crewai"):
        prompts, results = await _serve_concurrently(CrewAIAdapter(crew=crew))
    assert results == prompts
    assert crew.max_in_flight == 1
    assert crew.kickoffs == len(prompts)
    warnings = [r for r in caplog.records if "one at a time" in r.getMessage()]
    assert len(warnings) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "attrs",
    [
        {"knowledge_sources": ["source"]},
        {"max_rpm": 10},
        {"agents": [SimpleNamespace(max_rpm=10)]},
        {"manager_agent": SimpleNamespace(max_rpm=10)},
        {"agents": [SimpleNamespace(mcps=["https://example.test/mcp"])]},
        {"manager_agent": SimpleNamespace(mcps=["https://example.test/mcp"])},
        {"agents": [SimpleNamespace(apps=["gmail"])]},
        {"manager_agent": SimpleNamespace(apps=["gmail"])},
        {"agents": [SimpleNamespace(from_repository="org/agent")]},
        {"manager_agent": SimpleNamespace(from_repository="org/agent")},
        {"memory": True},
        {"short_term_memory": object()},
        {"long_term_memory": object()},
        {"entity_memory": object()},
        {"external_memory": object()},
    ],
    ids=[
        "knowledge-sources",
        "crew-max-rpm",
        "agent-max-rpm",
        "manager-max-rpm",
        "agent-mcps",
        "manager-mcps",
        "agent-apps",
        "manager-apps",
        "agent-from-repository",
        "manager-from-repository",
        "memory",
        "short-term-memory",
        "long-term-memory",
        "entity-memory",
        "external-memory",
    ],
)
async def test_crews_that_cannot_be_copied_faithfully_are_serialised(attrs):
    crew = CountingCopyCrew(**attrs)
    prompts, results = await _serve_concurrently(CrewAIAdapter(crew=crew))
    assert results == prompts
    assert crew.max_in_flight == 1
    assert crew.copy_calls == 0


@pytest.mark.asyncio
async def test_copy_that_raises_serialises_and_is_not_retried():
    class RaisingCopyCrew(CountingCopyCrew):
        def copy(self):
            self.copy_calls += 1
            raise ValueError("cannot deep copy")

    crew = RaisingCopyCrew()
    adapter = CrewAIAdapter(crew=crew)
    assert await adapter.invoke("first") == "first"
    assert crew.copy_calls == 1
    assert await adapter.invoke("second") == "second"
    assert crew.copy_calls == 1
    assert crew.kickoffs == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("returns", ["other-type", "self"])
async def test_unusable_copy_result_serialises_on_the_original(returns):
    class WrongCopyCrew(CountingCopyCrew):
        def copy(self):
            return self if returns == "self" else InFlightCrew()

    crew = WrongCopyCrew()
    prompts, results = await _serve_concurrently(CrewAIAdapter(crew=crew))
    assert results == prompts
    assert crew.kickoffs == len(prompts)
    assert crew.max_in_flight == 1


@pytest.mark.asyncio
async def test_fallback_lock_is_released_after_a_timeout():
    class HangsOnRequestCrew(InFlightCrew):
        async def kickoff_async(self, inputs=None):
            if inputs["inputs"] == "hang":
                await asyncio.sleep(10)
            return await super().kickoff_async(inputs=inputs)

    adapter = CrewAIAdapter(crew=HangsOnRequestCrew(), timeout=5)
    # Switches the adapter to the fallback, so the short timeout covers only the kickoff.
    assert await adapter.invoke("warm-up") == "warm-up"

    adapter.timeout = 0.05
    with pytest.raises(TimeoutError, match="timed out"):
        await adapter.invoke("hang")

    adapter.timeout = 5
    assert await adapter.invoke("second") == "second"


@pytest.mark.asyncio
async def test_request_queued_on_the_lock_times_out():
    started = asyncio.Event()
    release = asyncio.Event()

    class BlockingCrew(InFlightCrew):
        async def kickoff_async(self, inputs=None):
            self.kickoffs += 1
            started.set()
            await release.wait()
            return inputs["inputs"]

    crew = BlockingCrew()
    adapter = CrewAIAdapter(crew=crew, timeout=5)
    first = asyncio.create_task(adapter.invoke("first"))
    await asyncio.wait_for(started.wait(), 2)

    adapter.timeout = 0.02
    with pytest.raises(TimeoutError, match=r"CrewAI crew timed out after 0\.02 seconds"):
        await adapter.invoke("queued")

    release.set()
    assert await asyncio.wait_for(first, 2) == "first"
    assert crew.kickoffs == 1


@pytest.mark.asyncio
async def test_attribute_error_in_copied_crew_does_not_rerun_it():
    class BrokenCrew:
        def __init__(self, log=None):
            self.log = [] if log is None else log

        def copy(self):
            return BrokenCrew(self.log)

        async def kickoff_async(self, inputs=None):
            self.log.append("kickoff_async")
            raise AttributeError("raised inside the crew")

        def kickoff(self, inputs=None):
            self.log.append("kickoff")
            return "must not run"

    crew = BrokenCrew()
    with pytest.raises(AttributeError, match="raised inside the crew"):
        await CrewAIAdapter(crew=crew).invoke("hi")
    assert crew.log == ["kickoff_async"]


# ──── Test: _copy_crew against the real package (copy only, never kickoff) ────


@pytest.fixture
def real_crewai(tmp_path, monkeypatch):
    """Keep CrewAI offline and away from the real home directory."""
    monkeypatch.setenv("OTEL_SDK_DISABLED", "true")
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    crewai = pytest.importorskip("crewai")
    from crewai.llms.base_llm import BaseLLM

    from a2a_adapter.integrations.crewai import _copy_crew

    class FakeLLM(BaseLLM):
        def call(self, *args, **kwargs):
            raise AssertionError("these tests never run the crew")

    llm = FakeLLM(model="fake-model")

    def agent(role="{inputs} role", **kwargs):
        return crewai.Agent(
            role=role, goal="{inputs} goal", backstory="{inputs} story", llm=llm, **kwargs
        )

    def task(assignee, **kwargs):
        return crewai.Task(
            description="Do {inputs}", expected_output="{inputs} out", agent=assignee, **kwargs
        )

    return SimpleNamespace(module=crewai, agent=agent, task=task, copy=_copy_crew)


def test_copy_of_a_crew_that_already_ran_keeps_its_placeholders(real_crewai):
    first, second = real_crewai.agent("{inputs} first"), real_crewai.agent("{inputs} second")
    tasks = [real_crewai.task(second, output_file="{inputs}.md"), real_crewai.task(first)]
    crew = real_crewai.module.Crew(agents=[first, second], tasks=tasks)
    for task in tasks:
        task.interpolate_inputs_and_add_conversation_history({"inputs": "earlier"})
    for agent in crew.agents:
        agent.interpolate_inputs({"inputs": "earlier"})
    assert tasks[0].description == "Do earlier"

    clone = real_crewai.copy(crew)

    assert clone is not crew
    assert clone.tasks[0].description == "Do {inputs}"
    assert clone.tasks[0].expected_output == "{inputs} out"
    assert clone.tasks[0].output_file == "{inputs}.md"
    assert clone.agents[0].role == "{inputs} first"
    assert clone.agents[0].goal == "{inputs} goal"
    assert clone.agents[0].backstory == "{inputs} story"
    assert clone.tasks[0].agent is clone.agents[1]
    assert clone.tasks[1].agent is clone.agents[0]
    assert tasks[0].description == "Do earlier"


def test_memory_field_names_match_the_real_crew(real_crewai):
    from a2a_adapter.integrations.crewai import _MEMORY_FIELDS

    assert set(_MEMORY_FIELDS) <= set(real_crewai.module.Crew.model_fields)


def test_agent_field_names_match_the_real_agent(real_crewai):
    fields = real_crewai.module.Agent.model_fields
    assert {"max_rpm", "mcps", "apps", "from_repository"} <= set(fields)


def test_task_context_survives_the_copy(real_crewai):
    agent = real_crewai.agent()
    source = real_crewai.task(agent)
    follow_up = real_crewai.module.Task(
        description="Use it for {inputs}", expected_output="done", agent=agent, context=[source]
    )
    crew = real_crewai.module.Crew(agents=[agent], tasks=[source, follow_up])

    clone = real_crewai.copy(crew)

    assert clone.tasks[1].context[0] is clone.tasks[0]


def test_tasks_with_identical_text_and_context_are_not_copied(real_crewai):
    """Crew.copy maps context by a hash of the text, so it would pick the wrong twin."""
    agent = real_crewai.agent()
    first, twin = real_crewai.task(agent), real_crewai.task(agent)
    follow_up = real_crewai.module.Task(
        description="Use it", expected_output="done", agent=agent, context=[first]
    )
    crew = real_crewai.module.Crew(agents=[agent], tasks=[first, twin, follow_up])
    assert real_crewai.copy(crew) is None


def test_copying_forgets_finished_task_spans(real_crewai):
    """CrewAI's listener never drops these keys, and each one pins a whole copy."""
    from crewai.events.event_listener import event_listener

    agent = real_crewai.agent()
    finished, running = real_crewai.task(agent), real_crewai.task(agent)
    crew = real_crewai.module.Crew(agents=[agent], tasks=[real_crewai.task(agent)])
    event_listener.execution_spans[finished] = None
    event_listener.execution_spans[running] = span = object()
    try:
        assert real_crewai.copy(crew) is not None
        assert finished not in event_listener.execution_spans
        assert event_listener.execution_spans[running] is span
    finally:
        event_listener.execution_spans.pop(finished, None)
        event_listener.execution_spans.pop(running, None)


def test_crew_subclass_is_not_copied(real_crewai):
    class MyCrew(real_crewai.module.Crew):
        pass

    agent = real_crewai.agent()
    crew = MyCrew(agents=[agent], tasks=[real_crewai.task(agent)])
    assert real_crewai.copy(crew) is None


def test_agents_sharing_a_role_are_not_copied(real_crewai):
    first, second = real_crewai.agent("same"), real_crewai.agent("same")
    crew = real_crewai.module.Crew(
        agents=[first, second], tasks=[real_crewai.task(first), real_crewai.task(second)]
    )
    assert real_crewai.copy(crew) is None


@pytest.mark.parametrize(
    "kwargs",
    [{"mcps": ["https://example.test/mcp"]}, {"apps": ["gmail"]}],
    ids=["mcps", "apps"],
)
def test_agents_with_excluded_tool_sources_are_not_copied(real_crewai, kwargs):
    agent = real_crewai.agent(**kwargs)
    crew = real_crewai.module.Crew(agents=[agent], tasks=[real_crewai.task(agent)])
    assert real_crewai.copy(crew) is None
