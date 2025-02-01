import os

from forecasting_tools.forecasting.forecast_bots.official_bots.q1_template_bot import (
    Q1TemplateBot,
)
from forecasting_tools.forecasting.helpers.asknews_searcher import (
    AskNewsSearcher,
)
from forecasting_tools.forecasting.questions_and_reports.questions import (
    MetaculusQuestion,
)


class LaylapsBot(Q1TemplateBot):
    async def run_research(self, question: MetaculusQuestion) -> str:
        research = []
        if os.getenv("ASKNEWS_CLIENT_ID") and os.getenv("ASKNEWS_SECRET"):
            research.append(
                AskNewsSearcher.get_formatted_news(question.question_text)
            )
        if os.getenv("EXA_API_KEY"):
            research.append(
                await self._call_exa_smart_searcher(question.question_text)
            )
        if os.getenv("PERPLEXITY_API_KEY"):
            research.append(
                await self._call_perplexity(question.question_text)
            )
        else:
            raise ValueError("No API key provided")

        concat_research = ""
        for i, re in enumerate(research):
            concat_research += f"Research Source {i+1}:\n{re}\n"
        return concat_research
