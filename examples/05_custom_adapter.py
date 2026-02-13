"""
Example: Custom Adapter — Two Approaches

Shows two ways to create a custom A2A agent:
1. Subclass BaseA2AAdapter (full control)
2. Use CallableAdapter (simpler)

Both become full A2A servers in 3 lines.

Usage:
    python examples/05_custom_adapter.py           # Approach 1: subclass
    python examples/05_custom_adapter.py callable   # Approach 2: callable
"""

import sys


# ── Approach 1: Subclass BaseA2AAdapter ──

def run_custom_subclass():
    from a2a_adapter import BaseA2AAdapter, AdapterMetadata, serve_agent

    class SentimentAdapter(BaseA2AAdapter):
        async def invoke(self, user_input: str, context_id: str | None = None) -> str:
            positive = ["good", "great", "excellent", "happy", "love", "wonderful"]
            negative = ["bad", "terrible", "awful", "sad", "hate", "horrible"]
            text = user_input.lower()
            pos = sum(1 for w in positive if w in text)
            neg = sum(1 for w in negative if w in text)

            if pos > neg:
                sentiment = "POSITIVE"
            elif neg > pos:
                sentiment = "NEGATIVE"
            else:
                sentiment = "NEUTRAL"

            return f"Sentiment: {sentiment}\nConfidence: {min(0.6 + max(pos, neg) * 0.1, 0.95):.0%}\nText: {user_input}"

        def get_metadata(self) -> AdapterMetadata:
            return AdapterMetadata(
                name="Sentiment Analyzer",
                description="Analyzes the sentiment of text messages",
                skills=[{"id": "sentiment", "name": "Analyze Sentiment", "description": "Detect positive/negative/neutral sentiment"}],
            )

    serve_agent(SentimentAdapter(), port=8003)


# ── Approach 2: Use CallableAdapter ──

def run_callable():
    from a2a_adapter import CallableAdapter, serve_agent

    async def sentiment_analyzer(inputs):
        text = inputs.get("message", "").lower()
        positive = ["good", "great", "excellent", "happy", "love"]
        negative = ["bad", "terrible", "awful", "sad", "hate"]
        pos = sum(1 for w in positive if w in text)
        neg = sum(1 for w in negative if w in text)
        sentiment = "POSITIVE" if pos > neg else "NEGATIVE" if neg > pos else "NEUTRAL"
        return f"Sentiment: {sentiment} — {text}"

    adapter = CallableAdapter(
        func=sentiment_analyzer,
        name="Sentiment Analyzer (Callable)",
        description="Analyzes sentiment using a simple function",
    )

    serve_agent(adapter, port=8003)


if __name__ == "__main__":
    approach = sys.argv[1] if len(sys.argv) > 1 else "subclass"

    if approach == "callable":
        print("Starting Callable Sentiment Analyzer on port 8003...")
        run_callable()
    else:
        print("Starting Custom Sentiment Analyzer on port 8003...")
        run_custom_subclass()
