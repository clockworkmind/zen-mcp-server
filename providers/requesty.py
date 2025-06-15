"""Requesty.ai provider implementation."""

import logging
from typing import Optional

from .base import (
    ModelCapabilities,
    ModelResponse,
    ProviderType,
    RangeTemperatureConstraint,
)
from .openai_compatible import OpenAICompatibleProvider
from .requesty_registry import RequestyModelRegistry


class RequestyProvider(OpenAICompatibleProvider):
    """Requesty.ai unified API provider.

    Requesty provides access to multiple AI models through a single API endpoint.
    See https://requesty.ai for available models and pricing.
    """

    FRIENDLY_NAME = "Requesty"

    # Custom headers if needed (Requesty may not require special headers like OpenRouter)
    DEFAULT_HEADERS = {}

    # Model registry for managing configurations and aliases
    _registry: Optional[RequestyModelRegistry] = None

    def __init__(self, api_key: str, **kwargs):
        """Initialize Requesty provider.

        Args:
            api_key: Requesty API key
            **kwargs: Additional configuration
        """
        base_url = "https://router.requesty.ai/v1"
        super().__init__(api_key, base_url=base_url, **kwargs)

        # Initialize model registry
        if RequestyProvider._registry is None:
            RequestyProvider._registry = RequestyModelRegistry()

        # Log loaded models and aliases
        models = self._registry.list_models()
        aliases = self._registry.list_aliases()
        logging.info(f"Requesty loaded {len(models)} models with {len(aliases)} aliases")

    def _parse_allowed_models(self) -> None:
        """Override to disable environment-based allow-list.

        Requesty model access is controlled via the Requesty dashboard,
        not through environment variables.
        """
        return None

    def _resolve_model_name(self, model_name: str) -> str:
        """Resolve model aliases to Requesty model names.

        Args:
            model_name: Input model name or alias

        Returns:
            Resolved Requesty model name
        """
        # Try to resolve through registry
        config = self._registry.resolve(model_name)

        if config:
            if config.model_name != model_name:
                logging.info(f"Resolved model alias '{model_name}' to '{config.model_name}'")
            return config.model_name
        else:
            # If not found in registry, return as-is
            # This allows using models not in our config file
            logging.debug(f"Model '{model_name}' not found in registry, using as-is")
            return model_name

    def get_capabilities(self, model_name: str) -> ModelCapabilities:
        """Get capabilities for a model.

        Args:
            model_name: Name of the model (or alias)

        Returns:
            ModelCapabilities from registry or generic defaults
        """
        # Try to get from registry first
        capabilities = self._registry.get_capabilities(model_name)

        if capabilities:
            return capabilities
        else:
            # Resolve any potential aliases and create generic capabilities
            resolved_name = self._resolve_model_name(model_name)

            logging.debug(
                f"Using generic capabilities for '{resolved_name}' via Requesty. "
                "Consider adding to requesty_models.json for specific capabilities."
            )

            # Create generic capabilities with conservative defaults
            capabilities = ModelCapabilities(
                provider=ProviderType.REQUESTY,
                model_name=resolved_name,
                friendly_name=self.FRIENDLY_NAME,
                context_window=32_768,  # Conservative default context window
                supports_extended_thinking=False,
                supports_system_prompts=True,
                supports_streaming=True,
                supports_function_calling=False,
                temperature_constraint=RangeTemperatureConstraint(0.0, 2.0, 1.0),
            )

            # Mark as generic for validation purposes
            capabilities._is_generic = True

            return capabilities

    def get_provider_type(self) -> ProviderType:
        """Get the provider type."""
        return ProviderType.REQUESTY

    def validate_model_name(self, model_name: str) -> bool:
        """Validate if the model name is allowed.

        Requesty accepts any model name as it's a routing service.
        The actual validation happens on Requesty's end based on
        the API key's permissions.

        Args:
            model_name: Model name to validate

        Returns:
            Always True - Requesty handles validation internally
        """
        # Accept any model name - Requesty handles validation
        return True

    def generate_content(
        self,
        prompt: str,
        model_name: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_output_tokens: Optional[int] = None,
        **kwargs,
    ) -> ModelResponse:
        """Generate content using the Requesty API.

        Args:
            prompt: User prompt to send to the model
            model_name: Name of the model (or alias) to use
            system_prompt: Optional system prompt for model behavior
            temperature: Sampling temperature
            max_output_tokens: Maximum tokens to generate
            **kwargs: Additional provider-specific parameters

        Returns:
            ModelResponse with generated content and metadata
        """
        # Resolve model alias to actual Requesty model name
        resolved_model = self._resolve_model_name(model_name)

        # Call parent method with resolved model name
        return super().generate_content(
            prompt=prompt,
            model_name=resolved_model,
            system_prompt=system_prompt,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            **kwargs,
        )

    def supports_thinking_mode(self, model_name: str) -> bool:
        """Check if the model supports extended thinking mode.

        This depends on the specific model being used through Requesty.

        Args:
            model_name: Model to check

        Returns:
            Whether the model supports thinking mode
        """
        # Check in registry first
        config = self._registry.resolve(model_name)
        if config:
            return config.supports_extended_thinking

        # For unknown models, assume no thinking mode support
        return False