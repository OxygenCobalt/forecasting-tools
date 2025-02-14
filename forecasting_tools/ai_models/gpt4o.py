import logging
from typing import Final

from forecasting_tools.ai_models.model_interfaces.combined_llm_archetype import (
    CombinedLlmArchetype,
)

logger = logging.getLogger(__name__)


class Gpt4o(CombinedLlmArchetype):
    # See OpenAI Limit on the account dashboard for most up-to-date limit
    MODEL_NAME: Final[str] = "gpt-4o"
    REQUESTS_PER_PERIOD_LIMIT: Final[int] = 8_000
    REQUEST_PERIOD_IN_SECONDS: Final[int] = 60
    TIMEOUT_TIME: Final[int] = 40
    TOKENS_PER_PERIOD_LIMIT: Final[int] = 8_000_000
    TOKEN_PERIOD_IN_SECONDS: Final[int] = 60
