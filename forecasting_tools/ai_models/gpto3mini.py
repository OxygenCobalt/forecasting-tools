from typing import Any

from forecasting_tools.ai_models.ai_utils.response_types import (
    TextTokenCostResponse,
)
from forecasting_tools.ai_models.model_archetypes.openai_text_model import (
    OpenAiTextToTextModel,
)


class GptO3Mini(OpenAiTextToTextModel):
    # See OpenAI Limit on the account dashboard for most up-to-date limit
    MODEL_NAME: str = "o3-mini"
    REQUESTS_PER_PERIOD_LIMIT: int = 8_000
    REQUEST_PERIOD_IN_SECONDS: int = 60
    TIMEOUT_TIME: int = 120
    TOKENS_PER_PERIOD_LIMIT: int = 2_000_000
    TOKEN_PERIOD_IN_SECONDS: int = 60

    def __init__(
        self,
        *args: Any,
        system_prompt: str | None = None,
        **kwargs: Any,
    ):
        super().__init__(
            *args,
            use_developer_role=True,
            system_prompt=system_prompt,
            **kwargs,
        )

    @classmethod
    def _get_mock_return_for_direct_call_to_model_using_cheap_input(
        cls,
    ) -> TextTokenCostResponse:
        response = (
            super()._get_mock_return_for_direct_call_to_model_using_cheap_input()
        )
        response.total_tokens_used += 269  # Add reasoning tokens
        return response
