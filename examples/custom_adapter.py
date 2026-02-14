"""
Example: Custom Adapter as A2A Agent

Subclass BaseA2AAdapter, implement invoke(), get a full A2A server.
This example builds a simple keyword-based sentiment analyzer.

Usage:
    python examples/custom_adapter.py
"""

from a2a_adapter import BaseA2AAdapter, AdapterMetadata, serve_agent


class SentimentAdapter(BaseA2AAdapter):
    """Analyze sentiment of user messages using keyword matching."""

    POSITIVE = {"good", "great", "excellent", "happy", "love", "wonderful", "amazing"}
    NEGATIVE = {"bad", "terrible", "awful", "sad", "hate", "horrible", "poor"}

    async def invoke(self, user_input: str, context_id: str | None = None, **kwargs) -> str:
        words = set(user_input.lower().split())
        pos = len(words & self.POSITIVE)
        neg = len(words & self.NEGATIVE)

        if pos > neg:
            return f"Positive sentiment detected: \"{user_input}\""
        elif neg > pos:
            return f"Negative sentiment detected: \"{user_input}\""
        return f"Neutral sentiment detected: \"{user_input}\""

    def get_metadata(self) -> AdapterMetadata:
        return AdapterMetadata(
            name="Sentiment Analyzer",
            description="Analyzes the sentiment of text messages",
        )


serve_agent(SentimentAdapter(), port=8003)
