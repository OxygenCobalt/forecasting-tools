import textwrap
from datetime import datetime
from typing import TypeVar
from unittest.mock import Mock

from forecasting_tools.data_models.binary_report import BinaryReport
from forecasting_tools.data_models.forecast_report import ReasonedPrediction
from forecasting_tools.data_models.multiple_choice_report import (
    PredictedOption,
    PredictedOptionList,
)
from forecasting_tools.data_models.numeric_report import (
    NumericDistribution,
    Percentile,
)
from forecasting_tools.data_models.questions import (
    BinaryQuestion,
    MetaculusQuestion,
    MultipleChoiceQuestion,
    NumericQuestion,
)
from forecasting_tools.forecast_bots.forecast_bot import ForecastBot
from forecasting_tools.forecast_helpers.forecast_database_manager import (
    ForecastDatabaseManager,
)
from forecasting_tools.forecast_helpers.metaculus_api import MetaculusApi

T = TypeVar("T", bound=MetaculusQuestion)


class ForecastingTestManager:
    TOURNAMENT_SAFE_TO_PULL_AND_PUSH_TO = MetaculusApi.AI_WARMUP_TOURNAMENT_ID
    TOURNAMENT_WITH_MIXTURE_OF_OPEN_AND_NOT_OPEN = (
        MetaculusApi.CURRENT_QUARTERLY_CUP_ID
    )
    TOURNAMENT_WITH_MIX_OF_QUESTION_TYPES = (
        MetaculusApi.CURRENT_QUARTERLY_CUP_ID
    )
    TOURN_WITH_OPENNESS_AND_TYPE_VARIATIONS = (
        MetaculusApi.CURRENT_QUARTERLY_CUP_ID
    )

    @classmethod
    def get_fake_binary_question(
        cls, community_prediction: float | None = 0.7
    ) -> BinaryQuestion:
        question = BinaryQuestion(
            question_text="Will TikTok be banned in the US?",
            community_prediction_at_access_time=community_prediction,
        )
        return question

    @staticmethod
    def get_fake_forecast_report(
        community_prediction: float | None = 0.7, prediction: float = 0.5
    ) -> BinaryReport:
        return BinaryReport(
            question=ForecastingTestManager.get_fake_binary_question(
                community_prediction
            ),
            prediction=prediction,
            explanation=textwrap.dedent(
                """
                # Summary
                This is a test explanation

                ## Analysis
                ### Analysis 1
                This is a test analysis

                ### Analysis 2
                This is a test analysis
                #### Analysis 2.1
                This is a test analysis
                #### Analysis 2.2
                This is a test analysis
                - Conclusion 1
                - Conclusion 2

                # Conclusion
                This is a test conclusion
                - Conclusion 1
                - Conclusion 2
                """
            ),
            other_notes=None,
        )

    @staticmethod
    def mock_forecast_bot_run_forecast(
        subclass: type[ForecastBot], mocker: Mock
    ) -> Mock:
        test_binary_question = (
            ForecastingTestManager.get_fake_binary_question()
        )
        mock_function = mocker.patch(
            f"{subclass._run_individual_question_with_error_propagation.__module__}.{subclass._run_individual_question_with_error_propagation.__qualname__}"
        )
        assert isinstance(test_binary_question, BinaryQuestion)
        mock_function.return_value = (
            ForecastingTestManager.get_fake_forecast_report()
        )
        return mock_function

    @staticmethod
    def mock_add_forecast_report_to_database(mocker: Mock) -> Mock:
        mock_function = mocker.patch(
            f"{ForecastDatabaseManager.add_forecast_report_to_database.__module__}.{ForecastDatabaseManager.add_forecast_report_to_database.__qualname__}"
        )
        return mock_function

    @staticmethod
    def quarterly_cup_is_not_active() -> bool:
        # Quarterly cup is not active from the 1st to the 10th day of the quarter while the initial questions are being set
        current_date = datetime.now().date()
        day_of_month = current_date.day
        month = current_date.month

        is_first_month_of_quarter = month in [1, 4, 7, 10]
        is_first_10_days = day_of_month <= 10

        return is_first_month_of_quarter and is_first_10_days

    @staticmethod
    def mock_getting_benchmark_questions(mocker: Mock) -> Mock:
        mock_function = mocker.patch(
            f"{MetaculusApi.get_benchmark_questions.__module__}.{MetaculusApi.get_benchmark_questions.__qualname__}"
        )
        mock_function.return_value = [
            ForecastingTestManager.get_fake_binary_question()
        ]
        return mock_function


class MockBot(ForecastBot):
    research_calls: int = 0
    binary_calls: int = 0
    multiple_choice_calls: int = 0
    numeric_calls: int = 0
    summarize_calls: int = 0

    async def run_research(self, question: MetaculusQuestion) -> str:
        self.__class__.research_calls += 1
        return "Mock research"

    async def _run_forecast_on_binary(
        self, question: BinaryQuestion, research: str
    ) -> ReasonedPrediction[float]:
        self.__class__.binary_calls += 1
        return ReasonedPrediction(
            prediction_value=0.5,
            reasoning="Mock rationale",
        )

    async def _run_forecast_on_multiple_choice(
        self, question: MultipleChoiceQuestion, research: str
    ) -> ReasonedPrediction[PredictedOptionList]:
        self.__class__.multiple_choice_calls += 1

        # Create evenly distributed probabilities for each option
        num_options = len(question.options)
        probability_per_option = 1.0 / num_options

        predicted_options = [
            PredictedOption(
                option_name=option, probability=probability_per_option
            )
            for option in question.options
        ]

        return ReasonedPrediction(
            prediction_value=PredictedOptionList(
                predicted_options=predicted_options
            ),
            reasoning="Mock rationale",
        )

    async def _run_forecast_on_numeric(
        self, question: NumericQuestion, research: str
    ) -> ReasonedPrediction[NumericDistribution]:
        self.__class__.numeric_calls += 1

        # Create a simple distribution with 5 percentiles
        percentiles = [
            Percentile(value=question.lower_bound or 0, percentile=0.1),
            Percentile(
                value=(question.lower_bound or 0) * 0.75
                + (question.upper_bound or 100) * 0.25,
                percentile=0.25,
            ),
            Percentile(
                value=(
                    (question.lower_bound or 0) + (question.upper_bound or 100)
                )
                / 2,
                percentile=0.5,
            ),
            Percentile(
                value=(question.lower_bound or 0) * 0.25
                + (question.upper_bound or 100) * 0.75,
                percentile=0.75,
            ),
            Percentile(value=question.upper_bound or 100, percentile=0.9),
        ]

        mock_distribution = NumericDistribution(
            declared_percentiles=percentiles,
            open_upper_bound=question.open_upper_bound,
            open_lower_bound=question.open_lower_bound,
            upper_bound=question.upper_bound,
            lower_bound=question.lower_bound,
            zero_point=question.zero_point,
        )

        return ReasonedPrediction(
            prediction_value=mock_distribution,
            reasoning="Mock rationale",
        )

    async def summarize(
        self,
        question: MetaculusQuestion,
        research: str,
        prediction: ReasonedPrediction,
    ) -> str:
        self.__class__.summarize_calls += 1
        return "Mock summary"
