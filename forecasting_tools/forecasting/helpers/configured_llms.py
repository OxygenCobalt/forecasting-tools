from forecasting_tools.ai_models.gpt4ovision import (
    Gpt4oVision,
    Gpt4VisionInput,
)
from forecasting_tools.ai_models.gpto3mini import GptO3Mini
from forecasting_tools.ai_models.metaculus4o import Gpt4oMetaculusProxy


class BasicLlm(Gpt4oMetaculusProxy):
    # NOTE: If need be, you can force an API key here through OpenAI Client class variable
    pass


class AdvancedLlm(GptO3Mini):
    pass


class VisionLlm(Gpt4oVision):
    pass


class VisionData(Gpt4VisionInput):
    pass
