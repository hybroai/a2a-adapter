"""
Example: n8n Workflow as A2A Agent

Expose any n8n workflow as an A2A-compatible agent in 3 lines.

Prerequisites:
- A running n8n instance with a webhook trigger
- The webhook accepts POST with {"message": "..."} and returns a response

Usage:
    python examples/n8n_agent.py
"""

from a2a_adapter import N8nAdapter, serve_agent

adapter = N8nAdapter(
    webhook_url="http://localhost:5678/webhook/my-webhook",
    name="N8n Math Agent",
    description="Math operations powered by an n8n workflow",
    timeout=30,
    message_field="event",       # Field name your n8n webhook expects
    headers={
        # "Authorization": "Bearer YOUR_TOKEN",
    },
)

serve_agent(adapter, port=9000)
