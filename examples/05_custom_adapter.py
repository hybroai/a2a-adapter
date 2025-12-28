"""
Example: Custom Adapter Implementation

This example demonstrates how to create a custom adapter by:
1. Subclassing BaseAgentAdapter
2. Using the CallableAgentAdapter with a custom async function

Both approaches are shown for maximum flexibility.

Usage:
    python examples/05_custom_adapter.py
"""

import asyncio
from typing import Any, Dict

from a2a_adapter import BaseAgentAdapter, load_a2a_agent, serve_agent
from a2a.types import AgentCard, AgentCapabilities, Message, MessageSendParams, TextPart


# Approach 1: Subclass BaseAgentAdapter
class CustomAgentAdapter(BaseAgentAdapter):
    """
    Custom adapter that implements specific business logic.
    
    This example creates a simple sentiment analysis agent that
    analyzes the sentiment of user messages.
    """

    def __init__(self, model_name: str = "simple"):
        self.model_name = model_name

    async def to_framework(self, params: MessageSendParams) -> Dict[str, Any]:
        """Extract user message text."""
        user_message = ""
        if params.messages:
            last_message = params.messages[-1]
            if hasattr(last_message, "content"):
                if isinstance(last_message.content, list):
                    text_parts = [
                        item.text
                        for item in last_message.content
                        if hasattr(item, "text")
                    ]
                    user_message = " ".join(text_parts)
                elif isinstance(last_message.content, str):
                    user_message = last_message.content

        return {"text": user_message}

    async def call_framework(
        self, framework_input: Dict[str, Any], params: MessageSendParams
    ) -> Dict[str, Any]:
        """Perform sentiment analysis."""
        text = framework_input["text"].lower()
        
        # Simple sentiment analysis based on keywords
        positive_words = ["good", "great", "excellent", "happy", "love", "wonderful", "amazing"]
        negative_words = ["bad", "terrible", "awful", "sad", "hate", "horrible", "poor"]
        
        positive_count = sum(1 for word in positive_words if word in text)
        negative_count = sum(1 for word in negative_words if word in text)
        
        if positive_count > negative_count:
            sentiment = "positive"
            confidence = min(0.6 + (positive_count * 0.1), 0.95)
        elif negative_count > positive_count:
            sentiment = "negative"
            confidence = min(0.6 + (negative_count * 0.1), 0.95)
        else:
            sentiment = "neutral"
            confidence = 0.5
        
        return {
            "sentiment": sentiment,
            "confidence": confidence,
            "text": text,
        }

    async def from_framework(
        self, framework_output: Dict[str, Any], params: MessageSendParams
    ) -> Message:
        """Convert analysis results to A2A Message."""
        sentiment = framework_output["sentiment"]
        confidence = framework_output["confidence"]
        
        response_text = (
            f"Sentiment Analysis Results:\n"
            f"- Sentiment: {sentiment.upper()}\n"
            f"- Confidence: {confidence:.2%}\n"
            f"- Analyzed text: \"{framework_output['text'][:100]}...\""
        )
        
        return Message(
            role="assistant",
            content=[TextPart(type="text", text=response_text)],
        )


# Approach 2: Use CallableAgentAdapter with async function
async def sentiment_analyzer_function(inputs: Dict[str, Any]) -> str:
    """
    Custom async function that can be wrapped by CallableAgentAdapter.
    
    This is a simpler approach when you don't need full control over
    the adapter lifecycle.
    """
    text = inputs["message"].lower()
    
    # Simple sentiment analysis
    positive_words = ["good", "great", "excellent", "happy", "love"]
    negative_words = ["bad", "terrible", "awful", "sad", "hate"]
    
    positive_count = sum(1 for word in positive_words if word in text)
    negative_count = sum(1 for word in negative_words if word in text)
    
    if positive_count > negative_count:
        sentiment = "POSITIVE 😊"
    elif negative_count > positive_count:
        sentiment = "NEGATIVE 😞"
    else:
        sentiment = "NEUTRAL 😐"
    
    return f"Sentiment: {sentiment}\nMessage: {text}"


async def main_custom_subclass():
    """Example using custom BaseAgentAdapter subclass."""
    
    adapter = CustomAgentAdapter(model_name="simple")
    
    agent_card = AgentCard(
        name="Sentiment Analyzer (Custom Adapter)",
        description="Analyzes the sentiment of text messages using a custom adapter implementation.",
        capabilities=AgentCapabilities(
            streaming=False,
            async_execution=True,
        ),
    )
    
    print("Starting Custom Sentiment Analyzer on port 8003...")
    serve_agent(agent_card=agent_card, adapter=adapter, port=8003)


async def main_callable_adapter():
    """Example using CallableAgentAdapter."""
    
    adapter = await load_a2a_agent({
        "adapter": "callable",
        "callable": sentiment_analyzer_function,
        "supports_streaming": False,
    })
    
    agent_card = AgentCard(
        name="Sentiment Analyzer (Callable)",
        description="Analyzes the sentiment of text messages using a callable function adapter.",
        capabilities=AgentCapabilities(
            streaming=False,
            async_execution=True,
        ),
    )
    
    print("Starting Callable Sentiment Analyzer on port 8003...")
    serve_agent(agent_card=agent_card, adapter=adapter, port=8003)


if __name__ == "__main__":
    import sys
    
    print("Custom Adapter Examples")
    print("=" * 50)
    print("1. Custom BaseAgentAdapter subclass")
    print("2. CallableAgentAdapter with async function")
    print()
    
    choice = input("Choose example (1 or 2, default=1): ").strip() or "1"
    
    if choice == "1":
        asyncio.run(main_custom_subclass())
    elif choice == "2":
        asyncio.run(main_callable_adapter())
    else:
        print("Invalid choice")
        sys.exit(1)

