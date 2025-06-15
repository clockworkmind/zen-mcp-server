"""Tests for explicit provider selection syntax (provider:model)."""

import os
from unittest.mock import patch

from providers.base import ProviderType
from providers.custom import CustomProvider
from providers.gemini import GeminiModelProvider
from providers.openai import OpenAIModelProvider
from providers.openrouter import OpenRouterProvider
from providers.registry import ModelProviderRegistry
from providers.requesty import RequestyProvider


class TestExplicitProviderSelection:
    """Test cases for explicit provider selection using provider:model syntax."""

    def setup_method(self):
        """Clean up registry before each test."""
        # Store original state
        registry = ModelProviderRegistry()
        self._original_providers = registry._providers.copy()
        self._original_initialized = registry._initialized_providers.copy()

        # Clear registry
        registry._providers.clear()
        registry._initialized_providers.clear()

    def teardown_method(self):
        """Restore registry after each test."""
        registry = ModelProviderRegistry()
        registry._providers.clear()
        registry._initialized_providers.clear()
        registry._providers.update(self._original_providers)
        registry._initialized_providers.update(self._original_initialized)

    def test_explicit_requesty_selection(self):
        """Test explicit selection of Requesty provider."""
        with patch.dict(os.environ, {"REQUESTY_API_KEY": "test-key"}):
            ModelProviderRegistry.register_provider(ProviderType.REQUESTY, RequestyProvider)

            provider = ModelProviderRegistry.get_provider_for_model("requesty:claude")
            assert isinstance(provider, RequestyProvider)

            # Also test case insensitivity
            provider = ModelProviderRegistry.get_provider_for_model("REQUESTY:claude")
            assert isinstance(provider, RequestyProvider)

            provider = ModelProviderRegistry.get_provider_for_model("Requesty:gemini")
            assert isinstance(provider, RequestyProvider)

    def test_explicit_openrouter_selection(self):
        """Test explicit selection of OpenRouter provider."""
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
            ModelProviderRegistry.register_provider(ProviderType.OPENROUTER, OpenRouterProvider)

            provider = ModelProviderRegistry.get_provider_for_model("openrouter:claude")
            assert isinstance(provider, OpenRouterProvider)

            provider = ModelProviderRegistry.get_provider_for_model("openrouter:o3-mini")
            assert isinstance(provider, OpenRouterProvider)

    def test_explicit_google_selection(self):
        """Test explicit selection of Google/Gemini provider."""
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}):
            ModelProviderRegistry.register_provider(ProviderType.GOOGLE, GeminiModelProvider)

            provider = ModelProviderRegistry.get_provider_for_model("google:gemini-2.5-flash-preview-05-20")
            assert isinstance(provider, GeminiModelProvider)

            # Test with shorthand
            provider = ModelProviderRegistry.get_provider_for_model("google:flash")
            assert isinstance(provider, GeminiModelProvider)

            # Google provider validates specific model names
            provider = ModelProviderRegistry.get_provider_for_model("google:invalid-model")
            assert provider is None  # Should not match since model is invalid

    def test_explicit_openai_selection(self):
        """Test explicit selection of OpenAI provider."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            ModelProviderRegistry.register_provider(ProviderType.OPENAI, OpenAIModelProvider)

            provider = ModelProviderRegistry.get_provider_for_model("openai:o3-mini")
            assert isinstance(provider, OpenAIModelProvider)

    def test_explicit_custom_selection(self):
        """Test explicit selection of custom provider."""
        with patch.dict(os.environ, {
            "CUSTOM_API_KEY": "test-key",
            "CUSTOM_API_URL": "http://localhost:11434"
        }):
            # Register with factory function
            def custom_factory(api_key=None):
                return CustomProvider(api_key=api_key or "", base_url="http://localhost:11434")

            ModelProviderRegistry.register_provider(ProviderType.CUSTOM, custom_factory)

            provider = ModelProviderRegistry.get_provider_for_model("custom:llama3.2")
            assert isinstance(provider, CustomProvider)

    def test_invalid_provider_falls_back_to_priority(self):
        """Test that invalid provider names fall back to priority-based resolution."""
        with patch.dict(os.environ, {
            "OPENROUTER_API_KEY": "test-key",
            "REQUESTY_API_KEY": "test-key"
        }):
            ModelProviderRegistry.register_provider(ProviderType.OPENROUTER, OpenRouterProvider)
            ModelProviderRegistry.register_provider(ProviderType.REQUESTY, RequestyProvider)

            # Invalid provider should fall back to priority resolution
            provider = ModelProviderRegistry.get_provider_for_model("invalid:claude")
            # Should get OpenRouter since it's higher priority than Requesty
            assert isinstance(provider, OpenRouterProvider)

    def test_provider_not_registered(self):
        """Test explicit selection when provider is not registered."""
        # Don't register any providers
        provider = ModelProviderRegistry.get_provider_for_model("requesty:claude")
        assert provider is None

    def test_explicit_selection_with_complex_model_names(self):
        """Test explicit selection with model names containing special characters."""
        with patch.dict(os.environ, {"REQUESTY_API_KEY": "test-key"}):
            ModelProviderRegistry.register_provider(ProviderType.REQUESTY, RequestyProvider)

            # Model names with dots, dashes, etc.
            provider = ModelProviderRegistry.get_provider_for_model("requesty:gpt-4.1")
            assert isinstance(provider, RequestyProvider)

            provider = ModelProviderRegistry.get_provider_for_model("requesty:claude-4-sonnet")
            assert isinstance(provider, RequestyProvider)

    def test_multiple_colons_treated_as_regular_model(self):
        """Test that model names with multiple colons are treated as regular models."""
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
            ModelProviderRegistry.register_provider(ProviderType.OPENROUTER, OpenRouterProvider)

            # Multiple colons should not trigger explicit selection
            provider = ModelProviderRegistry.get_provider_for_model("foo:bar:baz")
            # Should use normal resolution and OpenRouter accepts any model
            assert isinstance(provider, OpenRouterProvider)

    def test_empty_provider_name(self):
        """Test handling of empty provider name."""
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
            ModelProviderRegistry.register_provider(ProviderType.OPENROUTER, OpenRouterProvider)

            # Empty provider name should fall back to normal resolution
            provider = ModelProviderRegistry.get_provider_for_model(":claude")
            assert isinstance(provider, OpenRouterProvider)

    def test_priority_when_explicit_provider_unavailable(self):
        """Test that unavailable explicit providers don't validate and fall back to priority."""
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
            ModelProviderRegistry.register_provider(ProviderType.OPENROUTER, OpenRouterProvider)

            # Requesty is not registered, so get_provider returns None
            # This causes the explicit selection to fail and fall back to priority resolution
            provider = ModelProviderRegistry.get_provider_for_model("requesty:claude")
            assert isinstance(provider, OpenRouterProvider)  # Falls back to priority resolution

    def test_whitespace_handling(self):
        """Test handling of whitespace in provider:model syntax."""
        with patch.dict(os.environ, {"REQUESTY_API_KEY": "test-key"}):
            ModelProviderRegistry.register_provider(ProviderType.REQUESTY, RequestyProvider)

            # These should still work (provider name is trimmed in lowercase conversion)
            provider = ModelProviderRegistry.get_provider_for_model("requesty :claude")
            assert isinstance(provider, RequestyProvider)

            provider = ModelProviderRegistry.get_provider_for_model(" requesty:claude")
            assert isinstance(provider, RequestyProvider)
