from forecasting_tools.forecasting.forecast_bots.community.laylaps import (
    LaylapsBot,
)


class LaylapsAskNewsBot(LaylapsBot):
    def __init__(self, **kwargs):
        super().__init__(**kwargs, research_used=["asknews"])


class LaylapsExaBot(LaylapsBot):
    def __init__(self, **kwargs):
        super().__init__(**kwargs, research_used=["exa"])


class LaylapsPerplexityBot(LaylapsBot):
    def __init__(self, **kwargs):
        super().__init__(**kwargs, research_used=["perplexity"])


class LaylapsAllResearchBot(LaylapsBot):
    def __init__(self, **kwargs):
        super().__init__(
            **kwargs, research_used=["asknews", "exa", "perplexity"]
        )
