from forecasting_tools.ai_models.gpto1 import GptO1
from forecasting_tools.ai_models.gpto3mini import GptO3Mini
from forecasting_tools.forecasting.forecast_bots.community.laylaps import (
    LaylapsBot,
)


class LaylapsO3Bot(LaylapsBot):
    FINAL_DECISION_LLM = GptO3Mini()


class LaylapsO1Bot(LaylapsBot):
    FINAL_DECISION_LLM = GptO1()

    # async def _run_forecast_on_binary(
    #     self, question: BinaryQuestion, research: str
    # ) -> ReasonedPrediction[float]:
    #     prompt = clean_indents(
    #         f"""
    #         You are a professional forecaster interviewing for a job.
    #         To succeed, you must provide an accurate forecast for a question.

    #         Good forecasters will:
    #         1. Give the status quo outcome extra weight, since the world changes slowly most of the time.
    #         2. Consider realistic scenarios that would result in a Yes or No outcome to account for unexpected outcomes.
    #         3. Ground predictions with as much research as possible.

    #         Include your rationale with your answer. The last thing you write is your final answer as "Probability: ZZ%", 0-100

    #         Your interview question is:
    #         {question.question_text}

    #         Question background:
    #         {question.background_info}

    #         This question's outcome will be determined by the specific criteria below. These criteria have not yet been satisfied:
    #         {question.resolution_criteria}

    #         {question.fine_print}

    #         Your research assistant says:
    #         {research}
    #         """
    #     )
    #     reasoning = await self.FINAL_DECISION_LLM.invoke(prompt)
    #     prediction = self._extract_forecast_from_binary_rationale(
    #         reasoning, max_prediction=1, min_prediction=0
    #     )
    #     return ReasonedPrediction(
    #         prediction_value=prediction, reasoning=reasoning
    #     )
