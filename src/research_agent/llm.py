"""Provider-agnostic LLM factory.

Uses LangChain's ``init_chat_model`` for unified model dispatch.
Model strings auto-resolve providers (``claude...`` -> Anthropic,
``gpt...`` -> OpenAI).  Override with prefix: ``ollama/llama3``.
"""

from __future__ import annotations

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel


def create_llm(
    model: str,
    *,
    max_tokens: int,
    temperature: float,
) -> BaseChatModel:
    """Create a chat model from a model identifier string.

    Args:
        model: Model name, optionally prefixed with provider
            (e.g., ``ollama/llama3``, ``openai/gpt-4o``).
        max_tokens: Maximum tokens in response.
        temperature: Sampling temperature.

    Returns:
        A ``BaseChatModel`` instance for the resolved provider.

    Examples:
        >>> llm = create_llm("claude-haiku-4-5-20251001", max_tokens=2048, temperature=0.0)
        >>> llm = create_llm("ollama/llama3", max_tokens=4096, temperature=0.1)
    """
    return init_chat_model(model, max_tokens=max_tokens, temperature=temperature)
