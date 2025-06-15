"""Tests for Requesty provider."""

import os
from unittest.mock import patch

from providers.base import ProviderType
from providers.registry import ModelProviderRegistry
from providers.requesty import RequestyProvider


class TestRequestyProvider:
    """Test cases for Requesty provider."""

    def test_provider_initialization(self):
        """Test Requesty provider initialization."""
        provider = RequestyProvider(api_key="test-key")
        assert provider.api_key == "test-key"
        assert provider.base_url == "https://router.requesty.ai/v1"
        assert provider.FRIENDLY_NAME == "Requesty"

    def test_custom_headers(self):
        """Test Requesty custom headers."""
        # Test default headers (if any)
        provider = RequestyProvider(api_key="test-key")
        # Requesty may not have custom headers like OpenRouter
        # But test the structure is there
        assert hasattr(provider, "DEFAULT_HEADERS")

    def test_model_validation(self):
        """Test model validation."""
        provider = RequestyProvider(api_key="test-key")

        # Should accept any model - Requesty handles validation
        assert provider.validate_model_name("gpt-4.1") is True
        assert provider.validate_model_name("claude-4-sonnet") is True
        assert provider.validate_model_name("gemini-2.5-pro-preview-05-06") is True
        assert provider.validate_model_name("gemini-2.5-flash-preview-05-20") is True
        assert provider.validate_model_name("any-model-name") is True
        assert provider.validate_model_name("GPT-4.1") is True
        assert provider.validate_model_name("unknown-model") is True

    def test_get_capabilities(self):
        """Test capability generation."""
        provider = RequestyProvider(api_key="test-key")

        # Test with a model in the registry (using alias)
        caps = provider.get_capabilities("claude")
        assert caps.provider == ProviderType.REQUESTY
        assert caps.model_name == "claude-4-sonnet"  # Resolved name
        assert caps.friendly_name == "Requesty"

        # Test with a model not in registry - should get generic capabilities
        caps = provider.get_capabilities("unknown-model")
        assert caps.provider == ProviderType.REQUESTY
        assert caps.model_name == "unknown-model"
        assert caps.context_window == 32_768  # Safe default
        assert hasattr(caps, "_is_generic") and caps._is_generic is True

    def test_model_alias_resolution(self):
        """Test model alias resolution."""
        provider = RequestyProvider(api_key="test-key")

        # Test alias resolution with the limited set of models
        assert provider._resolve_model_name("claude") == "claude-4-sonnet"
        assert provider._resolve_model_name("sonnet") == "claude-4-sonnet"
        assert provider._resolve_model_name("gemini-pro") == "gemini-2.5-pro-preview-05-06"
        assert provider._resolve_model_name("gemini-flash") == "gemini-2.5-flash-preview-05-20"
        assert provider._resolve_model_name("gpt-4.1") == "gpt-4.1"
        assert provider._resolve_model_name("gpt4.1") == "gpt-4.1"

        # Test case-insensitive
        assert provider._resolve_model_name("CLAUDE") == "claude-4-sonnet"
        assert provider._resolve_model_name("SONNET") == "claude-4-sonnet"
        assert provider._resolve_model_name("Gemini-Pro") == "gemini-2.5-pro-preview-05-06"
        assert provider._resolve_model_name("GPT-4.1") == "gpt-4.1"

        # Test direct model names (should pass through unchanged)
        assert provider._resolve_model_name("claude-4-sonnet") == "claude-4-sonnet"
        assert provider._resolve_model_name("gpt-4.1") == "gpt-4.1"

        # Test unknown models pass through
        assert provider._resolve_model_name("unknown-model") == "unknown-model"
        assert provider._resolve_model_name("custom/model-v2") == "custom/model-v2"

    def test_requesty_registration(self):
        """Test Requesty can be registered and retrieved."""
        with patch.dict(os.environ, {"REQUESTY_API_KEY": "test-key"}):
            # Clean up any existing registration
            try:
                ModelProviderRegistry.unregister_provider(ProviderType.REQUESTY)
            except KeyError:
                # Provider wasn't registered, that's fine
                pass

            # Register the provider
            ModelProviderRegistry.register_provider(ProviderType.REQUESTY, RequestyProvider)

            # Retrieve and verify
            provider = ModelProviderRegistry.get_provider(ProviderType.REQUESTY)
            assert provider is not None
            assert isinstance(provider, RequestyProvider)


class TestRequestyRegistry:
    """Test cases for Requesty model registry."""

    def test_registry_loading(self):
        """Test registry loads models from config."""
        from providers.requesty_registry import RequestyModelRegistry

        registry = RequestyModelRegistry()

        # Should have loaded models
        models = registry.list_models()
        assert len(models) > 0
        assert "claude-4-sonnet" in models
        assert "gpt-4.1" in models
        assert "gemini-2.5-pro-preview-05-06" in models
        assert "gemini-2.5-flash-preview-05-20" in models

        # Should have loaded aliases
        aliases = registry.list_aliases()
        assert len(aliases) > 0
        assert "claude" in aliases
        assert "sonnet" in aliases
        assert "gemini-pro" in aliases
        assert "gemini-flash" in aliases

    def test_registry_capabilities(self):
        """Test registry provides correct capabilities."""
        from providers.requesty_registry import RequestyModelRegistry

        registry = RequestyModelRegistry()

        # Test known model
        caps = registry.get_capabilities("claude")
        assert caps is not None
        assert caps.model_name == "claude-4-sonnet"
        assert caps.context_window == 200000  # Claude's context window

        # Test using full model name
        caps = registry.get_capabilities("claude-4-sonnet")
        assert caps is not None
        assert caps.model_name == "claude-4-sonnet"

        # Test unknown model
        caps = registry.get_capabilities("non-existent-model")
        assert caps is None

    def test_multiple_aliases_same_model(self):
        """Test multiple aliases pointing to same model."""
        from providers.requesty_registry import RequestyModelRegistry

        registry = RequestyModelRegistry()

        # All these should resolve to Claude Sonnet
        sonnet_aliases = ["sonnet", "claude", "claude-sonnet"]
        for alias in sonnet_aliases:
            config = registry.resolve(alias)
            assert config is not None
            assert config.model_name == "claude-4-sonnet"


class TestRequestyFunctionality:
    """Test Requesty-specific functionality."""

    def test_requesty_always_uses_correct_url(self):
        """Test that Requesty always uses the correct base URL."""
        provider = RequestyProvider(api_key="test-key")
        assert provider.base_url == "https://router.requesty.ai/v1"

        # Even if we try to change it, it should remain the Requesty URL
        # (This is a characteristic of the Requesty provider)
        provider.base_url = "http://example.com"  # Try to change it
        # But new instances should always use the correct URL
        provider2 = RequestyProvider(api_key="test-key")
        assert provider2.base_url == "https://router.requesty.ai/v1"

    def test_requesty_model_registry_initialized(self):
        """Test that model registry is properly initialized."""
        provider = RequestyProvider(api_key="test-key")

        # Registry should be initialized
        assert hasattr(provider, "_registry")
        assert provider._registry is not None
