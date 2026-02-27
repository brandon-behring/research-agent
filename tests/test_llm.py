"""Tests for the LLM factory."""

from __future__ import annotations

from unittest.mock import patch

from research_agent.llm import create_llm


class TestCreateLLM:
    """Tests for create_llm()."""

    @patch("research_agent.llm.init_chat_model")
    def test_returns_init_chat_model_result(self, mock_init) -> None:
        """create_llm returns whatever init_chat_model returns."""
        sentinel = object()
        mock_init.return_value = sentinel
        result = create_llm("test-model", max_tokens=100, temperature=0.5)
        mock_init.assert_called_once_with("test-model", max_tokens=100, temperature=0.5)
        assert result is sentinel

    @patch("research_agent.llm.init_chat_model")
    def test_passes_provider_prefix(self, mock_init) -> None:
        """Provider-prefixed model names are forwarded as-is."""
        mock_init.return_value = object()
        create_llm("ollama/llama3", max_tokens=4096, temperature=0.1)
        mock_init.assert_called_once_with("ollama/llama3", max_tokens=4096, temperature=0.1)

    @patch("research_agent.llm.init_chat_model")
    def test_anthropic_model_name(self, mock_init) -> None:
        """Anthropic model names pass through without modification."""
        mock_init.return_value = object()
        create_llm("claude-haiku-4-5-20251001", max_tokens=2048, temperature=0.0)
        mock_init.assert_called_once_with(
            "claude-haiku-4-5-20251001", max_tokens=2048, temperature=0.0
        )
