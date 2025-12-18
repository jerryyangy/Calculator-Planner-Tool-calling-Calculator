# Calculator-Planner-Tool-calling-Calculator


Agent-style: Planner + Tool-calling Calculator

This version uses an LLM to decide whether to use the calculator tool and returns structured results. It’s closer to typical LangGraph agent patterns.

### Highlights

Uses an LLM to plan and optionally call the calculator tool.
Tool execution is deterministic; LLM only orchestrates.
Finalizer converts the interaction into a clean answer.
