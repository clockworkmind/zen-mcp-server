"""Tests for Requesty model registry functionality."""

import json
import os
import tempfile

import pytest

from providers.base import ProviderType
from providers.requesty_registry import RequestyModelConfig, RequestyModelRegistry


class TestRequestyModelRegistry:
    """Test cases for Requesty model registry."""

    def test_registry_initialization(self):
        """Test registry initializes with default config."""
        registry = RequestyModelRegistry()

        # Should load models from default location
        assert len(registry.list_models()) > 0
        assert len(registry.list_aliases()) > 0

    def test_custom_config_path(self):
        """Test registry with custom config path."""
        # Create temporary config
        config_data = {
            "models": [
                {
                    "model_name": "test/model-1",
                    "aliases": ["test1", "t1"],
                    "context_window": 4096,
                    "supports_extended_thinking": False,
                    "supports_system_prompts": True,
                    "supports_streaming": True,
                    "supports_function_calling": True,
                    "supports_json_mode": True,
                }
            ]
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config_data, f)
            temp_path = f.name

        try:
            registry = RequestyModelRegistry(config_path=temp_path)
            assert len(registry.list_models()) == 1
            assert "test/model-1" in registry.list_models()
            assert "test1" in registry.list_aliases()
            assert "t1" in registry.list_aliases()
        finally:
            os.unlink(temp_path)

    def test_environment_variable_override(self):
        """Test REQUESTY_MODELS_PATH environment variable."""
        # Create custom config
        config_data = {
            "models": [
                {
                    "model_name": "env/model",
                    "aliases": ["envtest"],
                    "context_window": 8192,
                    "supports_extended_thinking": False,
                    "supports_system_prompts": True,
                    "supports_streaming": True,
                    "supports_function_calling": False,
                    "supports_json_mode": False,
                }
            ]
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config_data, f)
            temp_path = f.name

        try:
            # Set environment variable
            original_env = os.environ.get("REQUESTY_MODELS_CONFIG_PATH")
            os.environ["REQUESTY_MODELS_CONFIG_PATH"] = temp_path

            # Create registry without explicit path
            registry = RequestyModelRegistry()

            # Should load from environment path
            assert "env/model" in registry.list_models()
            assert "envtest" in registry.list_aliases()

        finally:
            # Restore environment
            if original_env is not None:
                os.environ["REQUESTY_MODELS_CONFIG_PATH"] = original_env
            else:
                del os.environ["REQUESTY_MODELS_CONFIG_PATH"]
            os.unlink(temp_path)

    def test_alias_resolution(self):
        """Test alias resolution functionality."""
        registry = RequestyModelRegistry()

        # Test various aliases for the limited model set
        test_cases = [
            ("claude", "claude-4-sonnet"),
            ("CLAUDE", "claude-4-sonnet"),  # Case insensitive
            ("sonnet", "claude-4-sonnet"),
            ("gemini-pro", "gemini-2.5-pro-preview-05-06"),
            ("gemini-flash", "gemini-2.5-flash-preview-05-20"),
            ("gpt4.1", "gpt-4.1"),
        ]

        for alias, expected_model in test_cases:
            config = registry.resolve(alias)
            assert config is not None, f"Failed to resolve alias '{alias}'"
            assert config.model_name == expected_model

    def test_direct_model_name_lookup(self):
        """Test looking up models by their full name."""
        registry = RequestyModelRegistry()

        # Should be able to look up by full model name
        config = registry.resolve("claude-4-sonnet")
        assert config is not None
        assert config.model_name == "claude-4-sonnet"

        config = registry.resolve("gpt-4.1")
        assert config is not None
        assert config.model_name == "gpt-4.1"

        config = registry.resolve("gemini-2.5-pro-preview-05-06")
        assert config is not None
        assert config.model_name == "gemini-2.5-pro-preview-05-06"

    def test_unknown_model_resolution(self):
        """Test resolution of unknown models."""
        registry = RequestyModelRegistry()

        # Unknown aliases should return None
        assert registry.resolve("unknown-alias") is None
        assert registry.resolve("") is None
        assert registry.resolve("non-existent") is None

    def test_model_capabilities_conversion(self):
        """Test conversion to ModelCapabilities."""
        registry = RequestyModelRegistry()

        config = registry.resolve("claude")
        assert config is not None

        caps = config.to_capabilities()
        assert caps.provider == ProviderType.REQUESTY
        assert caps.model_name == "claude-4-sonnet"
        assert caps.friendly_name == "Requesty"
        assert caps.context_window == 200000
        assert not caps.supports_extended_thinking

    def test_duplicate_alias_detection(self):
        """Test that duplicate aliases are detected."""
        config_data = {
            "models": [
                {
                    "model_name": "test/model-1",
                    "aliases": ["dupe"],
                    "context_window": 4096,
                    "supports_extended_thinking": False,
                    "supports_system_prompts": True,
                    "supports_streaming": True,
                    "supports_function_calling": True,
                    "supports_json_mode": True,
                },
                {
                    "model_name": "test/model-2",
                    "aliases": ["DUPE"],  # Same alias, different case
                    "context_window": 8192,
                    "supports_extended_thinking": False,
                    "supports_system_prompts": True,
                    "supports_streaming": True,
                    "supports_function_calling": True,
                    "supports_json_mode": True,
                },
            ]
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config_data, f)
            temp_path = f.name

        try:
            with pytest.raises(ValueError, match="Duplicate alias"):
                RequestyModelRegistry(config_path=temp_path)
        finally:
            os.unlink(temp_path)

    def test_missing_config_file(self):
        """Test behavior with missing config file."""
        # Use a non-existent path
        registry = RequestyModelRegistry(config_path="/non/existent/path.json")

        # Should initialize with empty maps
        assert len(registry.list_models()) == 0
        assert len(registry.list_aliases()) == 0
        assert registry.resolve("anything") is None

    def test_invalid_json_config(self):
        """Test handling of invalid JSON."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("{ invalid json }")
            temp_path = f.name

        try:
            registry = RequestyModelRegistry(config_path=temp_path)
            # Should handle gracefully and initialize empty
            assert len(registry.list_models()) == 0
            assert len(registry.list_aliases()) == 0
        finally:
            os.unlink(temp_path)

    def test_model_with_all_capabilities(self):
        """Test model with all capability flags."""
        config = RequestyModelConfig(
            model_name="test/full-featured",
            aliases=["full"],
            context_window=128000,
            supports_extended_thinking=True,
            supports_system_prompts=True,
            supports_streaming=True,
            supports_function_calling=True,
            supports_json_mode=True,
            description="Fully featured test model",
        )

        caps = config.to_capabilities()
        assert caps.context_window == 128000
        assert caps.supports_extended_thinking
        assert caps.supports_system_prompts
        assert caps.supports_streaming
        assert caps.supports_function_calling
        # Note: supports_json_mode is not in ModelCapabilities yet

    def test_gemini_models_configuration(self):
        """Test that Gemini models are properly configured."""
        registry = RequestyModelRegistry()

        # Test Gemini Pro
        config = registry.resolve("gemini-pro")
        assert config is not None
        assert config.model_name == "gemini-2.5-pro-preview-05-06"
        assert config.context_window == 1000000  # 1M context
        assert config.supports_extended_thinking  # Pro supports thinking mode

        # Test Gemini Flash
        config = registry.resolve("gemini-flash")
        assert config is not None
        assert config.model_name == "gemini-2.5-flash-preview-05-20"
        assert config.context_window == 1000000  # 1M context
        assert not config.supports_extended_thinking  # Flash doesn't support thinking

    def test_gpt_model_configuration(self):
        """Test that GPT-4.1 is properly configured."""
        registry = RequestyModelRegistry()

        config = registry.resolve("gpt4.1")
        assert config is not None
        assert config.model_name == "gpt-4.1"
        assert config.context_window == 128000
        assert not config.supports_extended_thinking
        assert config.supports_function_calling
