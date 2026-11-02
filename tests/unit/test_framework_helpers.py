"""Tests for the shared framework adapter helpers."""

import asyncio

import pytest

from a2a_adapter.integrations._framework import iter_with_timeout


class Source:
    def __init__(self, fail_on_close=False):
        self.closed = False
        self.fail_on_close = fail_on_close

    async def gen(self):
        try:
            for i in range(3):
                yield i
        finally:
            self.closed = True
            if self.fail_on_close:
                raise RuntimeError("teardown failed")


@pytest.mark.asyncio
async def test_closing_wrapper_early_closes_inner_iterator():
    source = Source()
    wrapper = iter_with_timeout(source.gen(), 5, "x")
    assert await anext(wrapper) == 0
    await wrapper.aclose()
    assert source.closed


class BadCloseIter:
    """Iterator whose aclose() fails; __anext__ raises or sleeps."""

    def __init__(self, error=None):
        self.error = error

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self.error is not None:
            raise self.error
        await asyncio.sleep(1)

    async def aclose(self):
        raise RuntimeError("teardown failed")


@pytest.mark.asyncio
async def test_cleanup_error_does_not_mask_stream_error():
    with pytest.raises(ValueError, match="stream failed"):
        _ = [i async for i in iter_with_timeout(BadCloseIter(ValueError("stream failed")), 5, "x")]


@pytest.mark.asyncio
async def test_cleanup_error_does_not_mask_timeout():
    with pytest.raises(TimeoutError, match="x timed out after 0.05 seconds"):
        _ = [i async for i in iter_with_timeout(BadCloseIter(), 0.05, "x")]


@pytest.mark.asyncio
async def test_cleanup_error_surfaces_after_normal_end():
    class Iter:
        def __aiter__(self):
            return self

        async def __anext__(self):
            raise StopAsyncIteration

        async def aclose(self):
            raise RuntimeError("teardown failed")

    with pytest.raises(RuntimeError, match="teardown failed"):
        _ = [i async for i in iter_with_timeout(Iter(), 5, "x")]
