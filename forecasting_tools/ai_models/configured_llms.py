from forecasting_tools.ai_models.deprecated_model_classes.gpt4ovision import (
    Gpt4oVision,
    Gpt4VisionInput,
)
from forecasting_tools.ai_models.gpto3mini import GptO3Mini

default_llms = {
    "basic": "gpt-4o",
    "advanced": "gpt-4o",
}


class BasicLlm(GptO3Mini):
    # NOTE: If need be, you can force an API key here through OpenAI Client class variable
    pass


class AdvancedLlm(GptO3Mini):
    pass


class VisionLlm(Gpt4oVision):
    pass


class VisionData(Gpt4VisionInput):
    pass
