import os
from asyncio import Semaphore

from forecasting_tools.forecasting.forecast_bots.official_bots.q1_template_bot import (
    Q1TemplateBot,
)
from forecasting_tools.forecasting.helpers.asknews_searcher import (
    AskNewsSearcher,
)
from forecasting_tools.forecasting.questions_and_reports.questions import (
    MetaculusQuestion,
)

_dumbAskNewsSema = Semaphore(1)


class LaylapsBot(Q1TemplateBot):
    def __init__(self, research_used: list[str] | None = None, **kwargs):
        super().__init__(**kwargs)
        self.research_used = research_used or ["exa"]
        print(f"Using research: {self.research_used}")

    async def run_research(self, question: MetaculusQuestion) -> str:
        research = []
        if (
            "asknews" in self.research_used
            and os.getenv("ASKNEWS_CLIENT_ID")
            and os.getenv("ASKNEWS_SECRET")
        ):
            research.append(await self._call_asknews(question.question_text))
        if "exa" in self.research_used and os.getenv("EXA_API_KEY"):
            research.append(
                await self._call_exa_smart_searcher(question.question_text)
            )
        if "perplexity" in self.research_used and os.getenv(
            "PERPLEXITY_API_KEY"
        ):
            research.append(
                await self._call_perplexity(question.question_text)
            )
        if len(research) > 1:
            return "\n\n".join(
                [
                    f"Research Source {i+1}:\n{re}"
                    for i, re in enumerate(research)
                ]
            )
        if research:
            return research[0]
        raise ValueError(
            "No research done, did configure a valid research provider and API key?"
        )

    async def _call_asknews(self, question_text: str) -> str:
        async with _dumbAskNewsSema:
            return await AskNewsSearcher.get_formatted_news(question_text)
